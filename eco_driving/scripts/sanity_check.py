"""Quick sanity checks: check_env, obs-bounds, degenerate-idle-vs-arrival reward check."""
import numpy as np
from gymnasium.utils.env_checker import check_env

from eco_driving.config import EnvConfig
from eco_driving.envs import EcoDrivingEnv


def run_check_env():
    env = EcoDrivingEnv(EnvConfig())
    check_env(env, warn=True, skip_render_check=True)
    print("check_env: PASS")


def run_bounds_stress(n_episodes=50):
    cfg = EnvConfig()
    env = EcoDrivingEnv(cfg)
    lo, hi = env.observation_space.low, env.observation_space.high
    rng = np.random.default_rng(123)
    max_abs = 0.0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        assert np.all(obs >= lo - 1e-6) and np.all(obs <= hi + 1e-6), f"OOB reset obs {obs}"
        terminated = truncated = False
        while not (terminated or truncated):
            a = env.action_space.sample()
            # bias toward accelerating half the time to exercise dynamics more
            if rng.uniform() < 0.5:
                a = np.array([abs(a[0])], dtype=np.float32)
            obs, r, terminated, truncated, info = env.step(a)
            assert np.all(obs >= lo - 1e-6) and np.all(obs <= hi + 1e-6), f"OOB step obs {obs}"
            max_abs = max(max_abs, np.max(np.abs(obs)))
    print(f"bounds stress ({n_episodes} random episodes): PASS, max|obs| = {max_abs:.4f}")


def run_idle_vs_arrival_check():
    """A policy that brakes to a stop and stays there must score clearly worse than
    one that keeps driving toward arrival -- i.e. 'stay stopped' must not be a
    dominant strategy. (Episodes start already moving at v0 in [v0_min, v0_max],
    so the degenerate policy has to actively brake to a stop, not just coast.)"""
    cfg = EnvConfig()

    def rollout(policy_fn, seed):
        env = EcoDrivingEnv(cfg)
        obs, info = env.reset(seed=seed)
        total_r = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            a = policy_fn(obs, env)
            obs, r, terminated, truncated, info = env.step(a)
            total_r += r
        return total_r, info

    # NOTE: the env's action is Delta-a (change in commanded accel), not absolute
    # acceleration -- these helper policies reason in absolute target accel and
    # convert via `target_a - env.a_prev`, same pattern used for the baseline
    # driver in evaluate.py.
    def stop_policy(obs, env):
        # brake hard to a stop, then hold at rest for the remainder of the episode
        v = obs[0] * env.cfg.v_max
        target_a = env.cfg.a_min if v > 0.05 else 0.0
        return np.array([target_a - env.a_prev], dtype=np.float32)

    def arrive_policy(obs, env):
        # crude bang-bang: accelerate unless very close to speed limit or a leader is close
        v = obs[0] * env.cfg.v_max
        gap = obs[8] * env.cfg.obs_gap_cap
        if gap < 15:
            target_a = env.cfg.a_min * 0.3
        elif v < env.cfg.v_max * 0.9:
            target_a = env.cfg.a_max * 0.6
        else:
            target_a = 0.0
        return np.array([target_a - env.a_prev], dtype=np.float32)

    stop_scores = [rollout(stop_policy, s)[0] for s in range(5)]
    arrive_scores = [rollout(arrive_policy, s)[0] for s in range(5)]
    print(f"brake-and-stop policy mean return: {np.mean(stop_scores):.2f}")
    print(f"arrive-ish policy mean return:     {np.mean(arrive_scores):.2f}")
    assert np.mean(arrive_scores) > np.mean(stop_scores), "Degenerate stop policy is not dominated!"
    print("stop-vs-arrival dominance check: PASS")


def run_idle_penalty_check():
    """w_idle: stopping without a valid reason is penalized; stopping legitimately
    (at a red light within stopping range, or close behind a present leader) is
    not. Guards against the "loiter at v=0" basin the idle penalty exists to remove."""
    from eco_driving.envs.signal import RED

    cfg = EnvConfig()

    # Case 1: stopped, no leader, far from the signal -> no valid reason, penalized.
    env = EcoDrivingEnv(cfg)
    env.reset(seed=0)
    env.leader.present = False
    env.x = 50.0
    env.v = 0.0
    env.a_prev = 0.0
    obs, r, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["idle_penalty"], "expected idle penalty when stopped with no leader/signal justification"

    # Case 2: stopped just short of the line while the signal is red -> justified, not penalized.
    env2 = EcoDrivingEnv(cfg)
    env2.reset(seed=0)
    env2.leader.present = False
    env2.x = cfg.signal_pos - 5.0
    env2.v = 0.0
    env2.a_prev = 0.0
    cycle_len = cfg.green_dur + cfg.yellow_dur + cfg.red_dur
    target_cycle_time = cfg.green_dur + cfg.yellow_dur + 1.0  # 1s into red
    env2.signal.offset = (target_cycle_time - env2.t) % cycle_len
    assert env2.signal.state(env2.t).phase == RED
    obs, r, term, trunc, info = env2.step(np.array([0.0], dtype=np.float32))
    assert not info["idle_penalty"], "should not penalize a legitimate stop at a red light"

    # Case 3: stopped close behind a present leader -> justified, not penalized.
    env3 = EcoDrivingEnv(cfg)
    env3.reset(seed=0)
    env3.leader.present = True
    env3.leader.x = 61.0  # gap = (61 - veh_length=5) - 50 = 6m < 2*desired_gap(=8m at v=0)
    env3.leader.v = 0.0
    env3.x = 50.0
    env3.v = 0.0
    env3.a_prev = 0.0
    obs, r, term, trunc, info = env3.step(np.array([0.0], dtype=np.float32))
    assert not info["idle_penalty"], "should not penalize a legitimate stop close behind the leader"

    print("idle-penalty (w_idle) justification check: PASS")


def run_action_mask_safety_check(n_seeds=100):
    """A3: 100-seed random-policy stress test of the safety mask (config.py
    mask_enabled). Confirms the mask makes red-running structurally
    unreachable (0 red-runs even under fully random actions) and that it does
    not itself cause deadlock (checked via timeout rate, NOT raw arrival rate --
    see note below).

    Note: raw arrival rate under a FULLY RANDOM policy is dominated by an
    orthogonal, pre-existing phenomenon unrelated to the mask: the delta-a
    action space's random-walk-correlated acceleration frequently rear-ends
    the leader (a leader is present in 80% of episodes). Verified by an A/B
    check: WITHOUT the mask, random-policy collisions are just as high
    (70/100) plus 15 red-runs on top (only 14/100 arrive); WITH the mask,
    collisions are similar (~74/100) but red-runs drop to exactly 0 and
    arrivals actually improve to 26/100. Zero timeouts in both conditions is
    what rules out "deadlock" specifically (a mask-induced failure to ever
    reach the line) -- collisions are a real but separate failure mode this
    check does not claim to fix.
    """
    cfg = EnvConfig()
    n_arrived = 0
    n_redrun = 0
    n_collision = 0
    n_timeout = 0
    for seed in range(n_seeds):
        env = EcoDrivingEnv(cfg)
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            a = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(a)
        if info.get("red_run"):
            n_redrun += 1
        if info.get("collision"):
            n_collision += 1
        if truncated:
            n_timeout += 1
        if terminated and not info.get("red_run") and not info.get("collision"):
            n_arrived += 1
    timeout_rate = n_timeout / n_seeds
    print(f"action-mask safety check ({n_seeds} random-policy seeds): "
          f"arrived={n_arrived}/{n_seeds}  red_run={n_redrun}  "
          f"collision={n_collision}  timeout={n_timeout}")
    assert n_redrun == 0, f"safety mask failed to prevent red-running ({n_redrun} red-runs)"
    assert timeout_rate <= 0.05, f"safety mask appears to cause deadlock ({timeout_rate:.0%} timeout rate)"
    print("action-mask safety check (0 red-runs, no deadlock): PASS")


def run_nonzero_displacement_check(n_episodes=10, n_steps=20):
    """A short random-action rollout must produce nonzero net displacement --
    guards against the env silently regressing to a v0=0 dead-stop start where a
    'never move' policy could become a stable local optimum for training."""
    cfg = EnvConfig()
    rng = np.random.default_rng(7)
    for ep in range(n_episodes):
        env = EcoDrivingEnv(cfg)
        obs, info = env.reset(seed=1000 + ep)
        x0 = env.x
        assert x0 == 0.0
        assert env.v > 0.0, "episode should start already in motion (v0_min > 0)"
        for _ in range(n_steps):
            a = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        assert env.x > x0, f"no net displacement after {n_steps} random steps (ep {ep})"
    print(f"nonzero-displacement check ({n_episodes} episodes): PASS")


if __name__ == "__main__":
    run_check_env()
    run_bounds_stress()
    run_idle_vs_arrival_check()
    run_idle_penalty_check()
    run_action_mask_safety_check()
    run_nonzero_displacement_check()
    print("\nAll sanity checks passed.")
