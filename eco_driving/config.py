"""Central configuration for the eco-driving environment, reward, signal, leader and training.

Every tunable constant lives here so behavior can be adjusted without touching env logic.
"""
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # --- Episode / route ---
    dt: float = 0.5                # s, control step
    t_max: float = 160.0           # s, episode time budget
    route_length: float = 800.0    # m

    # --- Vehicle limits ---
    v_max: float = 16.0            # m/s, speed limit
    a_min: float = -4.0            # m/s^2, hard braking
    a_max: float = 2.5             # m/s^2, max acceleration
    v0_min: float = 4.0            # m/s, episode start speed range (already approaching)
    v0_max: float = 12.0           # m/s

    # --- Vehicle physical parameters ---
    mass: float = 1500.0           # kg
    g: float = 9.81                # m/s^2
    c_roll: float = 0.015          # rolling resistance coeff
    rho_air: float = 1.225         # kg/m^3
    c_drag: float = 0.32           # drag coefficient
    frontal_area: float = 2.5      # m^2
    veh_length: float = 5.0        # m

    # --- Fuel model (power-based, ICE, calibratable not calibrated) ---
    b0: float = 0.20               # mL/s idle
    b1: float = 1.2e-4             # mL/s per W
    b2: float = 1.0e-9             # mL/s per W^2

    # --- Traffic signal (fixed-time SPaT) ---
    green_dur: float = 20.0
    yellow_dur: float = 3.0
    red_dur: float = 20.0
    signal_pos: float = 400.0      # m, stop line position along route
    max_offset: float = None       # if None -> full cycle length

    # --- Leader vehicle ---
    p_leader: float = 0.8
    leader_gap_min: float = 10.0
    leader_gap_max: float = 30.0
    leader_v_cruise_low: float = 0.5   # * v_max
    leader_v_cruise_high: float = 0.85  # * v_max
    leader_retarget_min: float = 3.0   # s
    leader_retarget_max: float = 8.0   # s
    leader_stop_prob_per_retarget: float = 0.35  # chance a retarget is a "near stop" event
    leader_near_stop_speed: float = 1.0  # m/s, "near stop" target speed
    sentinel_gap: float = 150.0    # m, reported gap when no leader present
    obs_gap_cap: float = 150.0     # m, clip for normalized gap observation
    obs_dist_signal_cap: float = 150.0  # m, clip for normalized distance-to-signal
    obs_time_cap: float = 45.0     # s, clip for time_to_change / time_until_green

    # --- Car-following desired gap (used by reward + IDM baseline) ---
    min_gap: float = 4.0           # m
    time_headway: float = 1.6      # s

    # --- Reward weights ---
    w_prog: float = 0.2
    w_fuel: float = 1.0
    w_time: float = 0.6
    w_jerk: float = 0.15
    w_gap: float = 0.12
    gap_err_clip_low: float = -15.0
    gap_err_clip_high: float = 30.0
    gap_err_scale: float = 10.0

    r_arrival: float = 40.0
    r_timeout: float = -30.0
    r_violation: float = -200.0

    # --- Misc ---
    seed: int = 0

    def __post_init__(self):
        if self.max_offset is None:
            self.max_offset = self.green_dur + self.yellow_dur + self.red_dur


@dataclass
class TrainConfig:
    total_timesteps: int = 400_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 8
    learning_rate: float = 3e-4
    buffer_size: int = 300_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 5_000
    ent_coef: str = "auto"
    # Default SAC "auto" target entropy is -action_dim = -1 for this 1-D action
    # space; on this task that can decay too fast for some seeds, collapsing
    # exploration before the policy escapes the "never move" local optimum (see
    # README pitfalls). -0.3 (less negative) keeps a higher entropy floor longer.
    target_entropy: float = -0.3
    net_arch: tuple = (256, 256)
    seeds: tuple = (0, 1, 2)
