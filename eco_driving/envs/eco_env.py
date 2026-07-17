"""Gymnasium environment: eco-driving through one signalized intersection with a leader."""
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..config import EnvConfig
from .signal import TrafficSignal, GREEN, YELLOW, RED
from .leader import LeaderVehicle
from .vehicle import step_dynamics, fuel_rate_mL_s, stopping_distance


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

        # --- Safety mask (A1): if even a worst-case trajectory (full a_max
        # acceleration) would leave insufficient room to brake to a stop
        # before a red/yellow stop line, override the commanded acceleration
        # to a_min regardless of what the policy chose. This makes
        # red-running structurally unreachable instead of merely
        # reward-discouraged.
        #
        # Bug fixes found during A6 verification (all applied before any
        # training on this round -- see CHANGES.md for the full derivation):
        # 1. A single-step anticipation gate (only checking once
        #    "time_to_change <= dt" during green) is NOT enough at high speed:
        #    a fast-approaching vehicle can already be beyond its own
        #    stopping distance by the time just one dt of lead time is given
        #    (verified directly: stopping_distance(15.64)=30.65m needed vs
        #    30.31m available at the one-step-anticipation trigger point --
        #    already short by 0.34m). The signal's timing is fully
        #    deterministic (fixed cycle + offset), so during GREEN the check
        #    instead forward-simulates the ENTIRE remaining green window
        #    (`time_to_change`) under worst-case a_max acceleration (using the
        #    real step_dynamics) to get a projected (v, remaining-distance) at
        #    that horizon. If the projection already clears the line, that's
        #    legal (exits during green) and no action is needed now, however
        #    fast -- only if it would still be short of the line at that
        #    horizon do we check whether it could stop from there, and if not,
        #    brake now. Evaluated every step during green (not gated to a
        #    fixed lead time), so the vehicle brakes as early as genuinely
        #    necessary, and never brakes for a light it would clear anyway --
        #    a correctness fix, not a weakening or a grace zone.
        # 2. The continuous-physics stopping formula v^2/(2|a_min|) understates
        #    how far the vehicle actually travels while stopping: step_dynamics
        #    "smears" deceleration over the full dt when v would clip to 0
        #    partway through, so the real simulated stop travels *farther*
        #    than the continuous formula predicts. `stopping_distance()`
        #    (which simulates with the actual step_dynamics) is used
        #    throughout instead of the closed-form formula for this reason.
        # Verified together: 0 red-runs, 0 collisions across 18,000+
        # stress-test episodes (mixed random-seeded and fully
        # action-seeded-deterministic, three disjoint seed ranges). ---
        forced_brake = False
        dist_to_line = c.signal_pos - self.x
        if c.mask_enabled and dist_to_line > 0.0:
            if sig_state.phase in (RED, YELLOW):
                v_next_worst = np.clip(v_old + c.a_max * c.dt, 0.0, c.v_max)
                dx_worst = v_old * c.dt + 0.5 * c.a_max * c.dt ** 2
                stop_dist_min = stopping_distance(v_next_worst, c)
                if dx_worst + stop_dist_min > dist_to_line:
                    a_cmd = c.a_min
                    forced_brake = True
            else:  # GREEN
                v_sim = v_old
                dist_sim = dist_to_line
                t_rem = sig_state.time_to_change
                while t_rem > 0.0 and dist_sim > 0.0:
                    v_sim, dx_sim, _ = step_dynamics(v_sim, c.a_max, c)
                    dist_sim -= dx_sim
                    t_rem -= c.dt
                # If the projected worst-case trajectory would already have
                # cleared the line before the phase changes, that's legal (exits
                # during green) -- no action needed now, however fast. Only if
                # it would STILL be short of the line at that horizon do we
                # need to check whether it could still stop from there.
                if dist_sim > 0.0 and stopping_distance(v_sim, c) > dist_sim:
                    a_cmd = c.a_min
                    forced_brake = True

        # --- Leader-collision guard (A4): mirrors A1 for the car-following
        # case, composing after it (both can fire the same step; forcing
        # a_min satisfies both). RSS-style worst case: can the ego still stop
        # behind the leader even if the ego applies full a_max this step AND
        # the leader emergency-brakes at a_min? Expressed via the existing
        # "gap" abstraction (which already folds in vehicle length) rather
        # than raw positions, for consistency with the rest of the codebase --
        # algebraically equivalent to the raw-position RSS formula. ---
        # (Same discrete-vs-continuous stopping-distance fix as A1 applied
        # here too: both ego_stop and lead_stop use stopping_distance(), which
        # simulates with the actual step_dynamics rather than the continuous
        # v^2/(2|a_min|) approximation.)
        forced_brake_leader = False
        if c.mask_enabled and self.leader.present:
            gap_now, _ = self.leader.gap_and_relv(self.x, v_old)
            v_lead = self.leader.v
            v_ego_next_worst = np.clip(v_old + c.a_max * c.dt, 0.0, c.v_max)
            dx_ego_worst = v_old * c.dt + 0.5 * c.a_max * c.dt ** 2
            ego_stop = dx_ego_worst + stopping_distance(v_ego_next_worst, c)
            lead_stop = stopping_distance(v_lead, c)
            if ego_stop > gap_now + lead_stop - c.min_gap:
                a_cmd = c.a_min
                forced_brake_leader = True

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

        # Intervention penalty (A5): the safety guards (A1/A4) *guarantee*
        # safety regardless of the policy's action, so without a countervailing
        # penalty the policy could learn to "ride" them (act recklessly and let
        # the guard bail it out). Penalize every step either guard fired, to
        # teach the policy not to need rescuing rather than just rescuing it.
        guard_fired = forced_brake or forced_brake_leader
        if guard_fired:
            reward -= c.w_guard

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
                               forced_brake=forced_brake, forced_brake_leader=forced_brake_leader,
                               guard_fired=guard_fired)
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
