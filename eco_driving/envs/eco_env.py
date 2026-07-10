"""Gymnasium environment: eco-driving through one signalized intersection with a leader."""
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..config import EnvConfig
from .signal import TrafficSignal, GREEN, YELLOW, RED
from .leader import LeaderVehicle
from .vehicle import step_dynamics, fuel_rate_mL_s


class EcoDrivingEnv(gym.Env):
    """11-D observation, 1-D acceleration action. See project README for full spec."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvConfig()
        c = self.cfg

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([c.a_min], dtype=np.float32),
                                        high=np.array([c.a_max], dtype=np.float32),
                                        dtype=np.float32)

        self.signal = TrafficSignal(c.green_dur, c.yellow_dur, c.red_dur)
        self.leader = LeaderVehicle(c)

        self.x = 0.0
        self.v = 0.0
        self.a_prev = 0.0
        self.t = 0.0

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        c = self.cfg
        rng = self.np_random

        self.x = 0.0
        # start already approaching the intersection (not from a dead stop): avoids
        # a one-time cold-start jerk cost and better matches the intended scenario.
        self.v = float(rng.uniform(c.v0_min, c.v0_max))
        self.a_prev = 0.0
        self.t = 0.0

        self.signal.reset(rng, max_offset=c.max_offset)
        self.leader.reset(rng, ego_x0=self.x, ego_v0=self.v)

        obs = self._get_obs()
        info = self._get_info(dx=0.0, fuel_mL=0.0, jerk=0.0)
        return obs, info

    def step(self, action):
        c = self.cfg
        rng = self.np_random
        a_cmd = float(np.asarray(action).reshape(-1)[0])

        v_old = self.v
        v_new, dx, a_eff = step_dynamics(v_old, a_cmd, c)
        x_new = self.x + dx

        fuel_rate = fuel_rate_mL_s(v_old, a_eff, c)
        fuel_mL = fuel_rate * c.dt
        jerk = (a_eff - self.a_prev) / c.dt

        # Signal phase evaluated at the start of this step (dt is small vs phase durations).
        sig_state = self.signal.state(self.t)
        crossed_line = (self.x < c.signal_pos) and (x_new >= c.signal_pos)
        red_run = crossed_line and sig_state.phase == RED

        # Advance leader using pre-step time/signal, then evaluate gap at new positions.
        self.leader.step(rng, self.t, self.signal)
        gap, rel_v = self.leader.gap_and_relv(x_new, v_new)
        collision = self.leader.present and gap <= 0.0

        t_new = self.t + c.dt

        # --- reward ---
        reward = c.w_prog * dx - c.w_fuel * fuel_mL - c.w_time * c.dt - c.w_jerk * jerk ** 2
        if self.leader.present:
            desired_gap = c.min_gap + c.time_headway * v_new
            gap_error = np.clip(gap - desired_gap, c.gap_err_clip_low, c.gap_err_clip_high) / c.gap_err_scale
            reward -= c.w_gap * gap_error ** 2

        terminated = False
        truncated = False
        violation = red_run or collision
        arrived = x_new >= c.route_length

        if violation:
            reward += c.r_violation
            terminated = True
        elif arrived:
            reward += c.r_arrival
            terminated = True
        elif t_new >= c.t_max:
            reward += c.r_timeout
            truncated = True

        # commit state
        self.x = x_new
        self.v = v_new
        self.a_prev = a_eff
        self.t = t_new

        obs = self._get_obs()
        info = self._get_info(dx=dx, fuel_mL=fuel_mL, jerk=jerk, gap=gap, rel_v=rel_v,
                               red_run=red_run, collision=collision, arrived=arrived,
                               sig_phase=sig_state.phase, a_eff=a_eff)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        c = self.cfg
        sig = self.signal.state(self.t)
        dist_to_signal = max(c.signal_pos - self.x, 0.0)
        is_green = 1.0 if sig.phase == GREEN else 0.0
        is_yellow = 1.0 if sig.phase == YELLOW else 0.0
        ttc = min(sig.time_to_change, c.obs_time_cap)
        tug = min(sig.time_until_green, c.obs_time_cap)

        gap, rel_v = self.leader.gap_and_relv(self.x, self.v)
        leader_present = 1.0 if self.leader.present else 0.0
        gap_n = min(max(gap, 0.0), c.obs_gap_cap)

        obs = np.array([
            self.v / c.v_max,
            np.clip(self.a_prev / max(abs(c.a_min), c.a_max), -1.0, 1.0),
            np.clip(dist_to_signal / c.obs_dist_signal_cap, 0.0, 1.0),
            is_green,
            is_yellow,
            ttc / c.obs_time_cap,
            tug / c.obs_time_cap,
            leader_present,
            gap_n / c.obs_gap_cap,
            np.clip(rel_v / c.v_max, -1.0, 1.0),
            np.clip((c.route_length - self.x) / c.route_length, -1.0, 1.0),
        ], dtype=np.float32)
        return obs

    def _get_info(self, **kwargs) -> dict:
        info = {
            "x": self.x, "v": self.v, "t": self.t, "a_prev": self.a_prev,
            "leader_present": self.leader.present,
        }
        info.update(kwargs)
        return info
