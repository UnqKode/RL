"""Leading vehicle model: stop-and-go cruise profile + signal-aware braking.

The leader must stay relevant to the ego vehicle (i.e., not simply accelerate to
v_max and drive away), so it repeatedly retargets a cruise speed in
[0.5*v_max, 0.85*v_max], occasionally aims for a near-stop, and always
decelerates for a red/yellow signal ahead within braking range.
"""
import numpy as np

from ..config import EnvConfig
from .signal import TrafficSignal, RED, YELLOW


class LeaderVehicle:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.present = False
        self.x = 0.0
        self.v = 0.0
        self.target_v = 0.0
        self.time_to_retarget = 0.0

    def reset(self, rng: np.random.Generator, ego_x0: float, ego_v0: float):
        cfg = self.cfg
        self.present = rng.uniform() < cfg.p_leader
        if not self.present:
            self.x = np.inf
            self.v = 0.0
            return
        gap0 = rng.uniform(cfg.leader_gap_min, cfg.leader_gap_max)
        self.x = ego_x0 + gap0
        self.v = ego_v0
        self._pick_new_target(rng)

    def _pick_new_target(self, rng: np.random.Generator):
        cfg = self.cfg
        if rng.uniform() < cfg.leader_stop_prob_per_retarget:
            self.target_v = cfg.leader_near_stop_speed
        else:
            self.target_v = rng.uniform(cfg.leader_v_cruise_low * cfg.v_max,
                                         cfg.leader_v_cruise_high * cfg.v_max)
        self.time_to_retarget = rng.uniform(cfg.leader_retarget_min, cfg.leader_retarget_max)

    def step(self, rng: np.random.Generator, t: float, signal: TrafficSignal):
        if not self.present:
            return
        cfg = self.cfg
        dt = cfg.dt

        self.time_to_retarget -= dt
        if self.time_to_retarget <= 0.0:
            self._pick_new_target(rng)

        target_v = self.target_v

        # Signal-aware braking: if signal ahead is red/yellow and within comfortable
        # braking range, override the cruise target to bring the leader to a stop
        # at the stop line.
        dist_to_signal = cfg.signal_pos - self.x
        sig_state = signal.state(t)
        if 0.0 < dist_to_signal < 120.0 and sig_state.phase in (RED, YELLOW):
            v_safe_stop = np.sqrt(max(0.0, 2.0 * abs(cfg.a_min) * 0.8 * dist_to_signal))
            target_v = min(target_v, v_safe_stop)

        # Simple proportional speed tracking, bounded by accel limits.
        a_cmd = np.clip((target_v - self.v) / 2.0, cfg.a_min, cfg.a_max)

        v_new = np.clip(self.v + a_cmd * dt, 0.0, cfg.v_max)
        a_eff = (v_new - self.v) / dt
        dx = self.v * dt + 0.5 * a_eff * dt * dt
        self.v = v_new
        self.x += max(dx, 0.0)

    def gap_and_relv(self, ego_x: float, ego_v: float):
        cfg = self.cfg
        if not self.present:
            return cfg.sentinel_gap, 0.0
        gap = (self.x - cfg.veh_length) - ego_x
        return gap, self.v - ego_v
