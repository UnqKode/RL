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
    """12-D observation, 1-D rate-limited-acceleration action. See project README
    for full spec.

    The action is Delta-a (change in commanded acceleration per step), NOT
    acceleration directly: `a_cmd = clip(a_prev + delta_a, a_min, a_max)`. This
    bounds jerk structurally. Callers that want to command an absolute target
    acceleration directly (e.g. the baseline controller, which must remain
    unaffected by this reparameterization) should pass `target_a - env.a_prev`
    as the action -- the resulting a_cmd is then exactly
    `clip(target_a, a_min, a_max)`, identical to the old raw-acceleration path,
    regardless of delta_a_max (the env does not clip the incoming delta itself,
    only the declared action_space -- used by the RL policy -- bounds it to
    [-delta_a_max, +delta_a_max]).
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvConfig()
        c = self.cfg

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-c.delta_a_max], dtype=np.float32),
                                        high=np.array([c.delta_a_max], dtype=np.float32),
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
        delta_a = float(np.asarray(action).reshape(-1)[0])
        a_cmd = np.clip(self.a_prev + delta_a, c.a_min, c.a_max)

        v_old = self.v

        # Signal phase evaluated at the start of this step (dt is small vs phase durations).
        sig_state = self.signal.state(self.t)

        # --- Safety mask (A1): if even a worst-case this-step action (full
        # a_max) would leave insufficient room to brake to a stop before a
        # red/yellow stop line, override the commanded acceleration to a_min
        # regardless of what the policy chose. This makes red-running
        # structurally unreachable instead of merely reward-discouraged. ---
        forced_brake = False
        dist_to_line = c.signal_pos - self.x
        if c.mask_enabled and sig_state.phase in (RED, YELLOW) and dist_to_line > 0.0:
            v_next_worst = np.clip(v_old + c.a_max * c.dt, 0.0, c.v_max)
            dx_worst = v_old * c.dt + 0.5 * c.a_max * c.dt ** 2
            stop_dist_min = v_next_worst ** 2 / (2 * abs(c.a_min))
            if dx_worst + stop_dist_min > dist_to_line:
                a_cmd = c.a_min
                forced_brake = True

        v_new, dx, a_eff = step_dynamics(v_old, a_cmd, c)
        x_new = self.x + dx

        fuel_rate = fuel_rate_mL_s(v_old, a_eff, c)
        fuel_mL = fuel_rate * c.dt
        jerk = (a_eff - self.a_prev) / c.dt

        crossed_line = (self.x < c.signal_pos) and (x_new >= c.signal_pos)
        red_run = crossed_line and sig_state.phase == RED

        # Advance leader using pre-step time/signal, then evaluate gap at new positions.
        self.leader.step(rng, self.t, self.signal)
        gap, rel_v = self.leader.gap_and_relv(x_new, v_new)
        collision = self.leader.present and gap <= 0.0

        t_new = self.t + c.dt

        # --- reward ---
        reward = c.w_prog * dx - c.w_fuel * fuel_mL - c.w_time * c.dt - c.w_jerk * jerk ** 2
        desired_gap = c.min_gap + c.time_headway * v_new
        if self.leader.present:
            gap_error = np.clip(gap - desired_gap, c.gap_err_clip_low, c.gap_err_clip_high) / c.gap_err_scale
            reward -= c.w_gap * gap_error ** 2

        # Idle penalty: being stopped is only "free" when there's a valid reason
        # (a red/yellow signal within stopping range, or a close leader) -- this
        # removes the comfortable "loiter at v=0" basin that a bare jerk penalty
        # otherwise creates (re-launching costs jerk; sitting still doesn't).
        dist_to_signal_new = c.signal_pos - x_new
        stopping_dist = (v_new ** 2) / (2 * abs(c.a_min) * c.idle_decel_frac) + c.idle_near_line
        phase_is_red_or_yellow = sig_state.phase in (RED, YELLOW)
        within_stopping_range = (0.0 < dist_to_signal_new < stopping_dist) or \
            (0.0 <= dist_to_signal_new < c.idle_near_line)
        signal_reason = phase_is_red_or_yellow and within_stopping_range
        leader_reason = self.leader.present and gap < c.idle_leader_gap_mult * desired_gap
        no_reason = not signal_reason and not leader_reason
        if v_new < c.idle_v_thresh and no_reason:
            reward -= c.w_idle

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
                               sig_phase=sig_state.phase, a_eff=a_eff,
                               idle_penalty=bool(v_new < c.idle_v_thresh and no_reason),
                               forced_brake=forced_brake)
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

        # GLOSA advisory speed: the glide speed that arrives exactly at the next
        # green (v_max if already green -- no need to glide).
        if sig.phase == GREEN:
            v_adv = c.v_max
        else:
            v_adv = np.clip(dist_to_signal / max(sig.time_until_green, 0.5), 0.0, c.v_max)

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
            np.clip(v_adv / c.v_max, 0.0, 1.0),
        ], dtype=np.float32)
        return obs

    def _get_info(self, **kwargs) -> dict:
        info = {
            "x": self.x, "v": self.v, "t": self.t, "a_prev": self.a_prev,
            "leader_present": self.leader.present,
        }
        info.update(kwargs)
        return info
