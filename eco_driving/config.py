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
    # The RL policy's action is a rate-of-change of acceleration (Delta-a), not
    # acceleration directly: a_cmd = clip(a_prev + delta_a, a_min, a_max). This
    # bounds jerk structurally (a smoothness guarantee "by construction") rather
    # than relying solely on the jerk penalty to discourage abrupt changes.
    delta_a_max: float = 1.5       # m/s^2 per step, max |change in commanded accel|

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
    # Round 3 (B1): raised 1.0 -> 1.5 to push harder for a genuine fuel
    # improvement now that red-running is prevented structurally (action
    # masking) rather than relying on reward balance alone (see CHANGES.md).
    w_fuel: float = 1.5
    w_time: float = 0.6
    # Reduced from 0.15: comfort is now largely guaranteed structurally by the
    # delta-a rate limit above, so the jerk weight no longer needs to be large
    # enough on its own to discourage abrupt changes -- a large w_jerk was what
    # made re-launching from a stop feel expensive and encouraged loitering.
    w_jerk: float = 0.05
    # ABLATION (see CHANGES.md): reverted C5 (was w_gap=0.20, clip_high=50) back
    # to the pre-C5 values to test whether the strengthened gap-following pull
    # was contributing to the red-light-running regression found after C1-C7.
    w_gap: float = 0.12
    gap_err_clip_low: float = -15.0
    gap_err_clip_high: float = 30.0
    gap_err_scale: float = 10.0

    # Penalty for stopping (v < 0.5) without a valid reason (not near a red/
    # yellow signal within stopping range, and not close behind a present
    # leader) -- removes the "stop forever" / "loiter at 0" comfortable basin.
    w_idle: float = 1.0
    idle_v_thresh: float = 0.5      # m/s, below this counts as "stopped"
    idle_near_line: float = 10.0    # m, always a valid reason to be stopped this close on red/yellow
    idle_decel_frac: float = 0.6    # fraction of |a_min| assumed available for the stopping-range calc
    idle_leader_gap_mult: float = 2.0  # gap < mult * desired_gap counts as a valid leader-following reason

    r_arrival: float = 40.0
    r_timeout: float = -30.0
    # Round 3 (A2): raised -200 -> -1000 as a belt-and-braces backstop. Primary
    # defense against red-running is now the action mask in EcoDrivingEnv.step
    # (see mask_* fields below), which should make this penalty almost never
    # trigger during training.
    r_violation: float = -1000.0

    # --- Round 3 (A1): safety mask -- if even a worst-case this-step action
    # (full a_max) would leave insufficient room to brake to a stop before a
    # red/yellow stop line, the commanded acceleration is overridden to a_min
    # regardless of what the policy chose. Makes red-running structurally
    # unreachable rather than merely discouraged by reward. ---
    mask_enabled: bool = True

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
    # Linear learning-rate decay (SB3 schedule input is "progress_remaining",
    # 1.0 at the start down to 0.0 at the end) to stabilize late training --
    # the previous constant 3e-4 let seeds visibly degrade in their final 10k
    # steps (see CHANGES.md).
    lr_start: float = 3e-4
    lr_end: float = 1e-4
    buffer_size: int = 300_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 5_000
    ent_coef: str = "auto"
    # -1.5 (more negative than SB3's default auto target of -action_dim = -1)
    # damps late-run entropy-driven policy drift now that the idle/loiter local
    # optimum is handled directly by w_idle rather than needing extra entropy
    # to escape it.
    target_entropy: float = -1.5
    net_arch: tuple = (256, 256)
    seeds: tuple = (0, 1, 2)
    # Per-seed override for learning_starts (fallback if a seed collapses into
    # the idle basin again despite w_idle/GLOSA/rate-limited actions).
    learning_starts_overrides: dict = field(default_factory=dict)
