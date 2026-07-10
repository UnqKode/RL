"""Conventional (non-eco) baseline driver: IDM car-following + late signal reaction.

This models the classic "rush up and idle" behavior: the driver ignores the signal
countdown until the red/yellow is within comfortable braking distance, then brakes
hard and holds at the stop line. It never anticipates or glides.
"""
import numpy as np

from ..config import EnvConfig
from ..envs.signal import RED, YELLOW


class IDMBaselineDriver:
    """Intelligent Driver Model with a late, reactive signal-braking overlay."""

    def __init__(self, cfg: EnvConfig,
                 idm_time_headway: float = 1.5,
                 idm_min_gap: float = 2.0,
                 idm_max_accel: float = 1.5,
                 idm_comfort_decel: float = 2.0,
                 idm_delta: float = 4.0,
                 late_reaction_margin: float = 1.15):
        self.cfg = cfg
        self.T = idm_time_headway
        self.s0 = idm_min_gap
        self.a_max_idm = min(idm_max_accel, cfg.a_max)
        self.b_comf = idm_comfort_decel
        self.delta = idm_delta
        # multiplier on the "just barely comfortable" braking distance before reacting;
        # >1 means the driver waits until later than strictly necessary (late reaction).
        self.late_margin = late_reaction_margin

    def act(self, v: float, gap: float, rel_v: float, leader_present: bool,
            dist_to_signal: float, sig_phase: str, time_to_change: float = 0.0) -> float:
        """dist_to_signal is SIGNED (signal_pos - x): positive = not yet crossed,
        negative/zero = already past the stop line (signal becomes irrelevant)."""
        cfg = self.cfg
        v0 = cfg.v_max

        # --- IDM free-flow + interaction term toward the leader ---
        free_term = 1.0 - (v / v0) ** self.delta
        if leader_present and gap < cfg.obs_gap_cap:
            gap_c = max(gap, 0.1)
            s_star = self.s0 + max(0.0, v * self.T + (v * (-rel_v)) / (2 * np.sqrt(self.a_max_idm * self.b_comf)))
            interaction_term = (s_star / gap_c) ** 2
        else:
            interaction_term = 0.0
        a_idm = self.a_max_idm * (free_term - interaction_term)

        # --- Late signal reaction: never anticipates during green; ignores the
        # countdown entirely and only reacts once the phase itself is yellow/red. ---
        # `brake_margin` > 1 makes the discrete-time stopping formula slightly
        # conservative so the vehicle actually reaches v=0 at/before the line
        # instead of creeping across it one small discrete step at a time.
        brake_margin = 1.3
        eps_d = 0.05  # avoid div-by-zero without masking the true (small) remaining distance
        # The smooth kinematic stopping formula a=-v^2/(2d) is asymptotic: v and d
        # shrink together but, under discrete dt steps, never exactly reach zero
        # together -- eventually the vehicle would creep across the line at a tiny
        # residual speed. A close-range "emergency stop" zone forces a real v=0
        # (the env clips v to >= 0, so full a_min from a small v hits exactly zero).
        emergency_zone = 3.0
        a_signal = cfg.a_max  # "no constraint" sentinel
        if dist_to_signal > 0.0:
            if sig_phase == RED:
                if v <= 0.0:
                    # already stopped: hold, never creep forward
                    a_signal = 0.0
                elif dist_to_signal <= emergency_zone:
                    a_signal = cfg.a_min
                else:
                    d = max(dist_to_signal, eps_d)
                    a_signal = max(-(v ** 2) / (2 * d) * brake_margin, cfg.a_min)
            elif sig_phase == YELLOW:
                # if current speed clears the line before red starts, no need to brake
                can_clear = dist_to_signal <= v * max(time_to_change, 1e-3)
                if not can_clear:
                    d = max(dist_to_signal, eps_d)
                    a_signal = max(-(v ** 2) / (2 * d) * brake_margin, cfg.a_min)

        a_cmd = min(a_idm, a_signal)
        return float(np.clip(a_cmd, cfg.a_min, cfg.a_max))
