"""Quick smoke test for the IDM baseline driver: confirm it arrives, never runs a
red light, and produces the expected 'rush up and idle' pattern (some full stops)."""
import numpy as np

from eco_driving.config import EnvConfig
from eco_driving.envs import EcoDrivingEnv
from eco_driving.envs.signal import GREEN, YELLOW
from eco_driving.baseline.idm_driver import IDMBaselineDriver


def rollout_baseline(seed: int, cfg: EnvConfig, driver: IDMBaselineDriver):
    env = EcoDrivingEnv(cfg)
    obs, info = env.reset(seed=seed)
    total_r = 0.0
    stops = 0
    max_jerk = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        dist_to_signal = cfg.signal_pos - env.x
        sig_state = env.signal.state(env.t)
        gap, rel_v = env.leader.gap_and_relv(env.x, env.v)
        a = driver.act(env.v, gap, rel_v, env.leader.present, dist_to_signal,
                        sig_state.phase, sig_state.time_to_change)
        obs, r, terminated, truncated, info = env.step(np.array([a], dtype=np.float32))
        total_r += r
        if info["v"] < 0.3:
            stops += 1
        max_jerk = max(max_jerk, abs(info["jerk"]))
    return dict(seed=seed, total_r=total_r, t=env.x, time=env.t, stops=stops,
                max_jerk=max_jerk, terminated=terminated, truncated=truncated,
                red_run=info.get("red_run", False), collision=info.get("collision", False),
                leader_present=env.leader.present)


if __name__ == "__main__":
    cfg = EnvConfig()
    driver = IDMBaselineDriver(cfg)
    for seed in range(8):
        res = rollout_baseline(seed, cfg, driver)
        print(res)
