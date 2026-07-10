"""Shared point-mass longitudinal dynamics and power-based fuel model."""
import numpy as np

from ..config import EnvConfig


def step_dynamics(v: float, a_cmd: float, cfg: EnvConfig):
    """Advance one control step. Returns (v_new, x_delta, a_eff).

    a_eff is the *effective* acceleration after the speed-limit clip and must be
    used for distance, fuel, and jerk -- never the raw command -- so the agent is
    not charged fuel/jerk for acceleration the speed limit prevented.
    """
    a_cmd = np.clip(a_cmd, cfg.a_min, cfg.a_max)
    v_new = np.clip(v + a_cmd * cfg.dt, 0.0, cfg.v_max)
    a_eff = (v_new - v) / cfg.dt
    dx = v * cfg.dt + 0.5 * a_eff * cfg.dt ** 2
    return v_new, dx, a_eff


def fuel_rate_mL_s(v: float, a_eff: float, cfg: EnvConfig) -> float:
    """Power-based ICE fuel-rate model. Deceleration only burns idle fuel (b0)."""
    power = (cfg.mass * a_eff
             + cfg.mass * cfg.g * cfg.c_roll
             + 0.5 * cfg.rho_air * cfg.c_drag * cfg.frontal_area * v ** 2) * v
    p_pos = max(0.0, power)
    return cfg.b0 + cfg.b1 * p_pos + cfg.b2 * p_pos ** 2
