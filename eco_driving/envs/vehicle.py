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


def stopping_distance(v0: float, cfg: EnvConfig) -> float:
    """Total forward distance traveled if a_min is applied every subsequent
    step from v0 until stopped, simulated with the SAME discrete step_dynamics
    the environment actually runs (not a continuous-physics v^2/(2a)
    approximation).

    This matters for safety-critical stopping-distance checks (see
    EcoDrivingEnv's A1/A4 guards): step_dynamics "smears" deceleration over the
    full dt when a vehicle would reach v=0 partway through a step (since a_eff
    is derived from clip(v + a*dt, 0, v_max), not from the true time-to-stop),
    which makes the vehicle travel *farther* in its final braking step than the
    continuous-physics formula predicts. A guard reasoning with the continuous
    formula can therefore trigger with what looks like enough margin and still
    let the vehicle cross a line it was trying to stop before.
    """
    v = v0
    total_dx = 0.0
    for _ in range(200):  # generous cap; stopping from v_max takes ~v_max/(|a_min|*dt) steps
        if v <= 0.0:
            break
        v, dx, _ = step_dynamics(v, cfg.a_min, cfg)
        total_dx += dx
    return total_dx


def fuel_rate_mL_s(v: float, a_eff: float, cfg: EnvConfig) -> float:
    """Power-based ICE fuel-rate model. Deceleration only burns idle fuel (b0)."""
    power = (cfg.mass * a_eff
             + cfg.mass * cfg.g * cfg.c_roll
             + 0.5 * cfg.rho_air * cfg.c_drag * cfg.frontal_area * v ** 2) * v
    p_pos = max(0.0, power)
    return cfg.b0 + cfg.b1 * p_pos + cfg.b2 * p_pos ** 2
