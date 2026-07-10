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
    """A policy that idles forever must score clearly worse than one that arrives."""
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

    def idle_policy(obs, env):
        return np.array([env.cfg.a_min * 0.0], dtype=np.float32)  # a=0 -> stays at rest

    def arrive_policy(obs, env):
        # crude bang-bang: accelerate unless very close to speed limit or a leader is close
        v = obs[0] * env.cfg.v_max
        gap = obs[8] * env.cfg.obs_gap_cap
        if gap < 15:
            return np.array([env.cfg.a_min * 0.3], dtype=np.float32)
        if v < env.cfg.v_max * 0.9:
            return np.array([env.cfg.a_max * 0.6], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    idle_scores = [rollout(idle_policy, s)[0] for s in range(5)]
    arrive_scores = [rollout(arrive_policy, s)[0] for s in range(5)]
    print(f"idle policy mean return:    {np.mean(idle_scores):.2f}")
    print(f"arrive-ish policy return:   {np.mean(arrive_scores):.2f}")
    assert np.mean(arrive_scores) > np.mean(idle_scores), "Degenerate idle policy is not dominated!"
    print("idle-vs-arrival dominance check: PASS")


if __name__ == "__main__":
    run_check_env()
    run_bounds_stress()
    run_idle_vs_arrival_check()
    print("\nAll sanity checks passed.")
