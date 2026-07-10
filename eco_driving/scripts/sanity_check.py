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

    def stop_policy(obs, env):
        # brake hard to a stop, then hold at rest for the remainder of the episode
        v = obs[0] * env.cfg.v_max
        if v > 0.05:
            return np.array([env.cfg.a_min], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    def arrive_policy(obs, env):
        # crude bang-bang: accelerate unless very close to speed limit or a leader is close
        v = obs[0] * env.cfg.v_max
        gap = obs[8] * env.cfg.obs_gap_cap
        if gap < 15:
            return np.array([env.cfg.a_min * 0.3], dtype=np.float32)
        if v < env.cfg.v_max * 0.9:
            return np.array([env.cfg.a_max * 0.6], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    stop_scores = [rollout(stop_policy, s)[0] for s in range(5)]
    arrive_scores = [rollout(arrive_policy, s)[0] for s in range(5)]
    print(f"brake-and-stop policy mean return: {np.mean(stop_scores):.2f}")
    print(f"arrive-ish policy mean return:     {np.mean(arrive_scores):.2f}")
    assert np.mean(arrive_scores) > np.mean(stop_scores), "Degenerate stop policy is not dominated!"
    print("stop-vs-arrival dominance check: PASS")


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
    run_nonzero_displacement_check()
    print("\nAll sanity checks passed.")
