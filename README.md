# Eco-Driving RL Agent (SAC) for a Signalized Intersection

A Soft Actor-Critic (SAC) policy that drives a point-mass vehicle through one
fixed-time signalized intersection while following a stop-and-go leader, trained
to balance **fuel, travel time, and comfort** jointly (not optimizing any one at
the expense of the others). Compared against a conventional IDM + late-braking
baseline that shows the classic "rush up and idle" pattern.

## Project layout

```
eco_driving/
  config.py            EnvConfig (env/reward constants) and TrainConfig (SAC hyperparams)
  envs/
    signal.py           fixed-time traffic signal (SPaT) with per-episode random offset
    leader.py           stop-and-go leading vehicle model
    vehicle.py          shared point-mass dynamics + power-based fuel model
    eco_env.py          gymnasium.Env: 11-D obs, 1-D accel action, reward, termination
  baseline/
    idm_driver.py       IDM car-following + late/reactive signal braking (non-eco baseline)
  scripts/
    sanity_check.py     check_env, obs-bounds stress test, idle-vs-arrival dominance check
    baseline_smoke.py    quick baseline rollout sanity check across seeds
    train.py             SAC training (Monitor -> DummyVecEnv -> VecNormalize -> EvalCallback)
    evaluate.py           deterministic evaluation vs. baseline + all plots
models/<seed>/           model_final.zip, best_model.zip, vecnormalize.pkl, evaluations.npz, logs
results/                  summary_metrics.csv, plots/
```

## How to run

```bash
# 1. sanity checks (fast, ~seconds)
python -m eco_driving.scripts.sanity_check
python -m eco_driving.scripts.baseline_smoke

# 2. train (one seed, or all 3 seeds sequentially)
python -m eco_driving.scripts.train --seed 0 --timesteps 400000
python -m eco_driving.scripts.train --all-seeds

# 3. evaluate the trained policies vs. the baseline on 10 fixed scenario seeds,
#    produce results/summary_metrics.csv and results/plots/*.png
python -m eco_driving.scripts.evaluate
```

Each training run saves, per seed, under `models/sac_seed<N>/`:
- `best_model.zip` — best checkpoint by deterministic `EvalCallback` reward (used for evaluation)
- `model_final.zip` — final policy at the end of training
- `vecnormalize.pkl` — the `VecNormalize` running statistics (obs mean/var) used to
  normalize observations for both training and downstream deterministic evaluation
- `evaluations.npz` — `EvalCallback`'s timesteps/results/ep_lengths, used for the learning-curve plot
- `train_metrics.csv` — mean reward/fuel/travel-time/stop-count logged every `eval_freq` steps

## Environment

Point-mass vehicle, `dt=0.5s`, `route_length=800m`, one fixed-time signal at `x=400m`
(`green=20s/yellow=3s/red=20s` + random per-episode offset), speed limit `16 m/s`,
`a ∈ [-4.0, 2.5] m/s²`. Each episode starts already approaching the intersection at
a random speed `v0 ~ Uniform(4, 12) m/s` (see pitfall #6) rather than from a dead
stop. A leading vehicle is present 80% of episodes, spawned
10–30m ahead; it repeatedly retargets a cruise speed in `[0.5, 0.85]·v_max`
(occasionally aiming for a near-stop) so it never simply free-flows away — the
policy must genuinely track it. The leader also brakes for the signal itself.

**Observation (11-D)**: ego speed, previous accel, distance-to-signal, is_green,
is_yellow, time-to-change, time-until-green, leader-present, gap, relative speed,
distance-remaining — all normalized into `[-1, 1]` (mostly `[0, 1]`).

**Action (1-D)**: desired acceleration, `Box(a_min, a_max)`.

**Dynamics**: the commanded acceleration is clipped, then the speed update is
clipped to `[0, v_max]`; the *effective* acceleration after that clip
(`a_eff = (v_new - v)/dt`) is what's used for distance, fuel, and jerk — never the
raw command. This matters because near the speed limit, a raw command like
`+2.0 m/s²` may produce an effective acceleration near zero; charging fuel/jerk
for the raw command would incorrectly penalize the agent for a limit it can't
control.

**Fuel**: power-based ICE model, `P = (m·a_eff + m·g·c_roll + 0.5·ρ·Cd·A·v²)·v`,
`fuel_rate = b0 + b1·max(0,P) + b2·max(0,P)²`. Braking only burns idle fuel `b0`.
These constants are **illustrative/representative, not calibrated to any real
vehicle** — they're chosen to make fuel, time, and comfort trade off against each
other in a believable way, not to reproduce measured consumption figures.

## Reward design

```
r =  + w_prog · dx  − w_fuel · fuel_mL − w_time · dt − w_jerk · jerk²  − w_gap · gap_error²
     (+ r_arrival on arrival / + r_timeout on timeout / + r_violation on red-run or collision)
```
Weights: `w_prog=0.2, w_fuel=1.0, w_time=0.6, w_jerk=0.15, w_gap=0.12`,
`r_arrival=+40, r_timeout=-30, r_violation=-200`.

The **gap term** is asymmetric and clipped: `desired_gap = min_gap + time_headway·v`
(4m + 1.6s·v), `gap_error = clip(gap − desired, −15, +30) / 10`, squared and
weighted. Clipping matters — an unclipped error explodes when the leader is far
ahead (its own free-flow speed can be far below `v_max`, but transient gaps of
100m+ are possible right after a leader re-accelerates) and would otherwise
dominate the whole reward and destabilize training.

### Pitfalls this design specifically avoids (and why they matter)

1. **Degenerate "stop forever."** With only `−fuel` and `−time` on a fixed-time
   episode, the reward-maximizing policy is to brake to a stop and never move
   again (near-zero fuel, and the episode just times out once). The progress
   term `w_prog·dx` plus the arrival bonus and timeout penalty make stopping
   clearly worse than arriving — `sanity_check.py::run_idle_vs_arrival_check`
   asserts this on every run (brake-and-stop policy scores ≈ **−309**, a crude
   "try to arrive" policy scores ≈ **−143**). This reward-level dominance is
   necessary but, as pitfall #5 below shows, not sufficient on its own — the
   *optimizer* also has to be able to see the gradient toward the better policy.
2. **Runaway-leader reward explosion.** Fixed via the clipped/asymmetric gap
   term above *and* by making the leader stop-and-go (never drifting away to
   `v_max`), so gap error stays in a sane range in practice, not just in the
   worst case.
3. **Observation out of declared bounds.** The most common instance is normalizing
   `a_prev` by `a_max` — since `|a_min| > a_max` (4.0 vs 2.5), a full-braking
   `a_prev = a_min` would normalize to **−1.6**, outside `[-1, 1]`. Fixed by
   normalizing by `max(|a_min|, a_max)`. Verified by `check_env` plus a 50-episode
   random-action bounds-stress test in `sanity_check.py`.
4. **Charging fuel/jerk for clipped acceleration.** Always compute fuel and jerk
   from `a_eff` (post speed-clip), never the raw commanded acceleration (see
   Dynamics above).
5. **`VecNormalize(norm_reward=True)` silently collapsing the learning signal to
   zero — the single biggest pitfall found during development.** The first full
   training run got stuck: every seed idled for the entire episode (timeout every
   time, fuel pinned at the ~32mL idle floor) with zero improvement through
   250k steps, exactly the "stop forever" failure mode pitfall #1 is supposed to
   prevent. A controlled ablation (leader on/off × `norm_reward` True/False, short
   budgets) isolated the cause: with `norm_reward=True`, both leader configurations
   were completely frozen (bit-identical eval metrics at every checkpoint); with
   `norm_reward=False`, both broke through to real driving. `VecNormalize`
   normalizes reward by a running estimate of the discounted return's std; with
   `gamma=0.99` over 320-step episodes that running estimate inflates enough to
   flatten the per-step gradient toward zero on this task, regardless of the
   specific reward composition. **Fix:** train with `norm_reward=False` (kept
   `norm_obs=True`, which is what mattered for network conditioning anyway). This
   is the one deviation from the literal "non-negotiable" tech-stack line in the
   original spec — justified by the ablation above, since the literal
   `norm_reward=True` config never learns at all on this task.
6. **Cold-start jerk cost from `v=0` at reset** (a secondary, compounding issue
   flagged during review): starting every episode at a dead stop means the very
   first nonzero action jumps `a_prev` from 0, producing a one-time jerk of
   `a/dt` and a `w_jerk·jerk²` cost that a stopped policy never has to pay. On its
   own this is a small one-time cost (easily outweighed by the arrival bonus once
   any gradient signal exists at all — item 5 above was the dominant effect), but
   it's still an avoidable, arguably unrealistic penalty and doesn't match the
   intended scenario ("approaching an intersection"). Fixed by sampling the
   episode's initial speed `v0 ~ Uniform(v0_min, v0_max) = Uniform(4, 12) m/s`
   instead of starting at rest. `sanity_check.py` now also asserts (a) a
   brake-to-a-stop-and-hold policy is dominated by a keep-driving policy, and
   (b) a short random-action rollout produces nonzero net displacement — both
   guard against this class of bug recurring silently.
7. **Baseline running red lights by accident** (a bug found and fixed during
   development, not part of the original design doc): an early version of the
   IDM baseline started braking as soon as the signal turned red/yellow using a
   "comfort deceleration" formula, *even when continuing at current speed would
   have legally cleared the intersection before red*. That unnecessary braking
   left the vehicle still short of the line when red arrived — a self-inflicted
   violation. Even after fixing that, the smooth kinematic stopping law
   `a = -v²/(2d)` is asymptotic (v and d shrink together but, under discrete
   0.5s steps, never hit exactly zero together) — the vehicle would creep the
   last few centimeters across the line at a residual crawl. Fixed with three
   changes: (a) only brake on yellow if the vehicle *cannot* clear the line
   before the phase turns red, (b) a margin on the stopping formula, and (c) a
   close-range "emergency stop" zone (last 3m) that always commands full
   braking (not the tapering formula) so the environment's own `v = clip(v, 0,
   ·)` forces an exact stop instead of an asymptotic crawl. Verified with a
   200-seed stress test (0 red-runs, 0 collisions).

## Algorithm

SAC, `MlpPolicy`, `net_arch=[256,256]`, `lr=3e-4`, `buffer_size=300_000`,
`batch_size=256`, `gamma=0.99`, `tau=0.005`, `train_freq=1`, `gradient_steps=1`,
`learning_starts=5_000`, `ent_coef="auto"`, `target_entropy=-0.3` (less negative
than SB3's default `-action_dim=-1`; see pitfall #8). Single training env
(`DummyVecEnv`) wrapped in `VecNormalize(norm_obs=True, norm_reward=False,
clip_obs=10)` — see pitfall #5 for why `norm_reward` is off. A separate
`VecNormalize(training=False, norm_reward=False)` eval env is scored by
`EvalCallback` every 10k steps (SB3 auto-syncs its normalization stats from the
training env); the best checkpoint by deterministic eval reward is kept.

**Only 2 of the intended 3 seeds are reported (seeds 0 and 1).** Seed 2 was
attempted three times — full 400k runs under both the default entropy target and
`target_entropy=-0.3`, plus a shortened diagnostic — and never escaped the
idle/timeout failure mode in any of them (still `travel_time≈160s` and fuel at
the idle floor even past 200k steps every time). This is a genuine, reproducible
finding about this task's sensitivity to SAC's initialization/seed, not a script
bug: seeds 0 and 1 broke through cleanly (some noisily, but cleanly) on their
first attempt each. See pitfall #8 for the likely connection to the "stall on
stop" issue below. `models/sac_seed2/` is kept as-is (three interrupted attempts'
worth of logs) as evidence for this finding rather than deleted.

## Evaluation

`evaluate.py` runs both the baseline and every trained seed's `best_model.zip`
deterministically (`predict(..., deterministic=True)`) on **10 fixed scenario
seeds** (500–509, disjoint from training) — same seed ⇒ identical signal offset
and leader behavior for both drivers, for a fair comparison. Deterministic
evaluation feeds observations through the saved `VecNormalize` obs statistics
directly (equivalent to `training=False, norm_reward=False` since we don't touch
the running stats or use the reward at eval time).

Reported per driver: travel time, total fuel (mL), stop-steps (`v < 0.3 m/s`),
max `|jerk|`, plus arrival/timeout/red-run/collision counts — averaged over the
10 scenarios, and for the policy, additionally averaged (mean ± std) across
seeds. Plots (`results/plots/`): speed-vs-position, speed-vs-time (signal phase
shaded), cumulative fuel, gap-vs-time (actual vs. desired headway) for a
leader-present scenario, and the learning curves.

### Actual results (seeds 0, 1; 10 fixed scenarios)

| | travel_time (s) | fuel (mL) | stop-steps | max\|jerk\| | arrived |
|---|---|---|---|---|---|
| baseline (IDM) | 93.25 ± 17.93 | 79.31 ± 7.18 | 0.60 ± 1.80 | 3.38 ± 0.36 | 10/10 |
| SAC seed 0 | 96.40 ± 21.53 | 82.64 ± 10.86 | 45.70 ± 30.91 | 2.96 ± 0.81 | 10/10 |
| SAC seed 1 | 104.25 ± 27.82 | 85.09 ± 11.13 | 63.00 ± 56.55 | 4.06 ± 1.27 | 9/10 (1 timeout) |
| SAC across seeds | 100.33 ± 3.92 | 83.87 ± 1.22 | 54.35 ± 8.65 | 3.51 ± 0.55 | 19/20 |

## Definition of done — actual status (honest assessment)

This project does **not** fully meet the definition of done as originally
specified, for reasons documented below rather than papered over:

- ✅ `check_env` passes; training runs end-to-end with saved artifacts and
  reproducible seeds; all 5 required plots are generated.
- ✅ Baseline is safe: 10/10 arrivals, 0 red-runs, 0 collisions (verified further
  on 200+ extra seeds during development).
- ❌ **"Deterministic eco-policy arrives in every eval scenario"** — seed 0 does
  (10/10); seed 1 does not (9/10, one timeout). See pitfall #8.
- ❌ **"Uses less fuel and is smoother (lower max|jerk|) than baseline at
  comparable travel time"** — travel time is comparable (within ~11s on
  average), and seed 0's jerk (2.96) *is* lower than baseline's (3.38), but fuel
  is higher, not lower (83.87 vs 79.31 mL across seeds), and jerk is worse on
  average once seed 1 (4.06) is included. Root cause: pitfall #8.
- ✅ **Gap tracking** — see `results/plots/gap_vs_time.png`; when actually
  following (not stalled), the policy's gap tracks the desired headway rather
  than free-flowing to the 150m sentinel.

### Pitfall #8: policy sometimes fails to resume from a full stop ("stall-on-stop")

Found by tracing individual evaluation scenarios, not anticipated by the
original design doc. Both seed 0 and seed 1 exhibit it; seed 1 hit it badly
enough to time out once.

**Symptom:** the policy brakes to `v=0` for a legitimate reason (e.g. shadowing
the leader's stop-and-go near-stop event), but then, even once the leader has
sped off far ahead (gap growing past 100m, 300m, 600m+) and no signal is forcing
a stop, the policy keeps commanding negative acceleration and stays parked at
`v=0` for tens of seconds before eventually resuming. In the worst observed case
(seed 1, scenario 500) it never resumed in time and the episode timed out; seed
0 hit the same pattern multiple times in the same trace but happened to resume
soon enough to still arrive (barely — 124s of a 160s budget). This inflates the
`stop-steps` metric well above baseline's (which rarely fully stops at all) and
is the likely reason fuel/jerk don't come out clearly ahead of baseline.

**Root-cause hypothesis (not yet fixed):** the gap-error term is deliberately
clipped (see Reward design) so that being 34m too far from the leader costs the
same as being 600m too far — necessary to prevent the reward exploding when the
leader is legitimately far away, but it also means there's no reward *gradient*
pulling the agent to resume progress once it's already stopped and the leader is
long gone; only the flat `w_time`/`w_prog` terms are left to do that job. This
specific state region — stopped, leader-irrelevantly-far, no red light nearby —
combined with `gap_present=1` still being set (the flag doesn't distinguish "far
but present" from "close") appears to have been under-visited or badly
reinforced during training, possibly for the same underlying reason seed 2 never
escaped the idle optimum at all (pitfall #5/#7): once stopped, escaping is a
low-density event in the replay buffer, and if early escape attempts from this
particular state pocket happened to end in a timeout during training, the critic
would have learned to undervalue resuming *specifically from this state*, even
though it correctly values resuming from other states. This was not resolved
within the compute budget spent on this task; a follow-up fix would likely
involve either loosening the gap-error clip's flat region, adding an explicit
"no justification to be stopped" penalty, or extending training well past 400k
steps with reward shaping targeted at this exact state region.
