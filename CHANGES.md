# Changes: attempt to beat the baseline (C1–C7)

This documents the changes applied per the task spec, what they fixed, and a new
regression discovered during evaluation that means the task's acceptance
criteria are **not yet met**. Reported honestly rather than shipped silently,
per the task's own instructions.

## What was changed

| # | Change | File(s) |
|---|---|---|
| C1 | Confirmed `evaluate.py` already loaded `best_model.zip` (added an explicit print) | `scripts/evaluate.py` |
| C2 | Added `w_idle=1.0` penalty for stopping (`v<0.5`) with no valid reason (not near a red/yellow within stopping range, not close behind a leader) | `config.py`, `envs/eco_env.py` |
| C3 | Action reparameterized to rate-limited `Δa ∈ [-1.5, 1.5]`; `a_cmd = clip(a_prev + Δa, a_min, a_max)`. `w_jerk` reduced 0.15 → 0.05. Baseline driver converts its absolute target accel via `delta = target_a - a_prev` (proven mathematically exact, see below) so its behavior is unaffected. | `config.py`, `envs/eco_env.py`, `scripts/evaluate.py`, `scripts/baseline_smoke.py`, `scripts/sanity_check.py` |
| C4 | Added GLOSA advisory-speed feature (12th obs dim): glide speed to arrive exactly at next green | `envs/eco_env.py` |
| C5 | Gap-error clip widened `+30 → +50`; `w_gap` raised `0.12 → 0.20` | `config.py` |
| C6 | Linear LR decay `3e-4 → 1e-4`; `target_entropy` changed `-0.3 → -1.5` | `config.py`, `scripts/train.py` |
| C7 | Retrained all 3 seeds from scratch, 400k steps each (fresh — obs/action space changed, old checkpoints incompatible) | — |

## Verification that the baseline is unaffected by C3

`rollout_baseline`'s `act_fn` computes `delta = target_a - env.a_prev` and passes
that to `env.step()`. Since the env computes
`a_cmd = clip(a_prev + delta, a_min, a_max) = clip(a_prev + (target_a - a_prev), a_min, a_max) = clip(target_a, a_min, a_max)`,
this is mathematically exact regardless of `delta_a_max` (the env does not clip
the incoming delta itself, only the RL policy's declared `action_space` does).
Verified numerically: max reconstruction mismatch across a full episode was
`1.1e-16` (floating-point noise). Baseline metrics are unchanged from before
this round: `travel_time=93.25s, fuel=79.31mL, stops=0.6, jerk=3.38, 10/10 arrived`.

## What C2 fixed: the "stall-on-stop" / loitering defect

The previous round's known limitation (policy sometimes freezes at `v=0` after a
legitimate stop and fails to resume even once nothing is blocking it) appears
**resolved**. Stop-steps dropped dramatically:

| | before (prior round) | after (this round) |
|---|---|---|
| seed 0 mean stop-steps | 45.7 ± 30.9 | **0.6 ± 1.8** |
| seed 1 mean stop-steps | 63.0 ± 56.6 | **2.1 ± 3.0** |
| seed 2 | (never converged) | **2.5 ± 5.0** |

seed 2 also converged cleanly this time (previous round: 3 full attempts, all
still stuck in the idle/timeout basin past 200k+ steps). This directly
corroborates the task's own prediction that C2 would remove that basin.

## NEW regression found during evaluation: red-light running

**This is the blocking issue.** All three retrained seeds now run red lights in
3–4 of the 10 fixed eval scenarios — a failure mode that was **absent** from
both the baseline (0/10 always) and the previous round's SAC policies (0/10
red-runs for seed 0 and seed 1). This was not anticipated by the C1–C7 spec.

Traced two scenarios directly (`models/sac_seed0`, scenarios 501 and 505):

- **Scenario 505**: at `t=35.5`, `v=10.48 m/s`, `dist_to_signal=87.8m`, phase
  yellow. Stopping distance needed at max braking is only ~14m — trivially
  achievable with 87.8m available. Instead, the policy *accelerates*
  (`a_cmd`: 0.97 → 1.63 → 1.35 → ... ) the moment the phase flips to red at
  `t=36.0`, and drives straight through at ~11 m/s, crossing the line 7.5s into
  the red phase. No braking attempt at all.
- **Scenario 501**: a subtler variant — the policy *does* slow to near-stop
  (`v≈2 m/s`) well before the line while still yellow (looks like correct
  anticipatory gliding), but then reverses and accelerates hard
  (`a_cmd` up to `2.50`, the max) exactly as the phase turns red, again driving
  through.

Both cases show the policy had every opportunity and enough information
(dist-to-signal, is_red, GLOSA advisory speed all in the observation) to stop
safely, but instead chose to accelerate at the exact yellow→red transition.

### Suspected cause (not confirmed, needs further investigation)

Two candidate contributors, not mutually exclusive:

1. **Training instability (criterion 7 also fails for all 3 seeds).** The
   `evaluations.npz` history shows all three seeds' last 3 eval checkpoints
   scoring substantially worse than their best (relative deviations of
   1.4–5.4x, well outside the 15% stability band). `best_model.zip` is chosen
   by peak mean eval reward over only 8 episodes — a small sample that may not
   include a red-light scenario, so a checkpoint that "peaked" on reward could
   still harbor an intermittent, rare-but-severe red-running bug that wasn't
   sampled during that particular eval. The lower `target_entropy=-1.5` (C6)
   was meant to *reduce* late-training drift, but the instability persisted or
   worsened regardless — this needs direct comparison against the previous
   round's (less negative target-entropy) stability, which did not exhibit
   red-running.
2. **Reward-balance shift from C5.** Raising `w_gap` (0.12→0.20) and widening
   the far-side clip (+30→+50) makes the continuous, every-step gap-following
   signal larger in magnitude and present on ~80% of steps (leader is present
   in 8/10 scenarios), versus the sparse (rare, episode-ending) `-200`
   violation penalty. In principle `-200` should dominate any single episode's
   return, but reward-frequency imbalance during *training* (dense small
   signal vs. rare huge signal) can bias the learned critic's gradient more
   than the raw magnitudes suggest — this is analogous to the `norm_reward`
   pitfall from the first round, where a technically-correct reward design
   still failed to produce the intended gradient in practice. Not yet isolated
   via ablation (would need a training run with C5 reverted, holding C2–C4/C6
   fixed, to test in isolation).

### What was NOT the cause

Verified the rate-limited action space (C3) is not a *mechanical* cause: in
scenario 505 the policy had 87.8m of room and 10+ steps before reaching the
line — even a 3-4-step ramp-up to full braking (worst case under
`delta_a_max=1.5`) would comfortably have stopped it. The policy simply chose
to accelerate instead; this is a decision failure, not a rate-limit reachability
failure.

## Current results (do not represent a passing state)

| | travel_time (s) | fuel (mL) | stop-steps | max\|jerk\| | arrived | red-runs |
|---|---|---|---|---|---|---|
| baseline | 93.25 | 79.31 | 0.6 | 3.38 | 10/10 | 0 |
| SAC seed 0 | 73.55 | 64.74 | 0.6 | 2.60 | 6/10 | **4** |
| SAC seed 1 | 77.80 | 72.57 | 2.1 | 3.52 | 7/10 | **3** |
| SAC seed 2 | 78.05 | 72.24 | 2.5 | 2.91 | 7/10 | **3** |

Paired fuel delta (arrived-only scenarios): seed0 −1.1%, seed1 +5.9%, seed2
+5.4% (mean +3.4% ± 3.2% — worse than baseline, not better, and the paired
sample is small/biased because it excludes the red-run scenarios where the
policy would likely have used less fuel by stopping less).

## Acceptance criteria: 5 of 7 fail

1. Arrival (≥9/10, zero red-runs): **FAIL** — all 3 seeds have 3-4 red-runs.
2. Stops (≤15 mean): **PASS** — all 3 seeds well under.
3. Fuel (≤-5% for ≥2/3 seeds, no seed >+2%): **FAIL** — 0/3 seeds ≤-5%, 2 seeds >+2%.
4. Time (within +10s of baseline): **PASS** — all 3 seeds faster, not just within budget.
5. Smoothness (mean max|jerk| ≤ baseline): **PASS** for seeds 0, 2; **FAIL** for seed 1 (3.52 > 3.38).
6. Following (no sustained >2.5x-desired gap excursion >10s): **PASS** for seed 0; **FAIL** for seeds 1, 2 (12.5s, 13.0s).
7. Stability (last 3 eval points within 15% of best): **FAIL** for all 3 seeds.

## Ablation result: C5 confirmed as the cause, but reveals fuel was never actually beaten

Reverted C5 only (`w_gap` 0.20→0.12, `gap_err_clip_high` 50→30), retrained seed 0
from scratch into `models/sac_seed0_ablation_noC5/`, evaluated on the same 10
scenarios:

**Safety is fully restored: 10/10 arrived, 0 red-runs, 0 collisions, 0 timeouts**
— confirming C5's strengthened gap-following pull was indeed the cause.

**But paired fuel delta vs. baseline is now +9.8% ± 16.7% (worse, not better):**

| scenario | baseline fuel | ablation fuel | delta | (had red-run with C5?) |
|---|---|---|---|---|
| 500 | 91.0 | 85.6 | −5.9% | no |
| 501 | 72.6 | 95.6 | **+31.7%** | **yes** |
| 502 | 69.3 | 66.9 | −3.5% | no |
| 503 | 84.2 | 86.7 | +3.0% | no |
| 504 | 75.7 | 95.6 | **+26.3%** | **yes** |
| 505 | 72.8 | 93.7 | **+28.7%** | **yes** |
| 506 | 78.7 | 81.9 | +4.1% | no |
| 507 | 90.9 | 118.3 | **+30.1%** | **yes** |
| 508 | 76.0 | 64.0 | −15.7% | no |
| 509 | 81.9 | 81.5 | −0.5% | no |

The four scenarios with the largest fuel increase are **exactly** the four
scenarios where the C5-enabled (unsafe) policy ran the red light. This is the
key insight: the earlier "the SAC policy uses less fuel" result for those
scenarios was an artifact of skipping the stop — a policy that safely brakes,
idles, and re-accelerates for a red light necessarily burns more fuel than one
that sails straight through it. **The apparent fuel win from the C1-C7 retrain
was not legitimate.** Once safety is restored, this seed uses *more* fuel than
baseline, not less — the original task goal (beat the baseline on fuel while
remaining safe) has not yet been achieved.

## Recommendation

Do not ship the C1-C7 (with-C5) checkpoints — confirmed unsafe (red-running).
The ablation checkpoint (`sac_seed0_ablation_noC5`) is safe but not yet better
than baseline on fuel. Achieving both simultaneously needs further work, e.g.:
tuning `w_fuel` upward now that the gap term is back to its safe weight, a
longer training budget, or a more targeted car-following reward that doesn't
reintroduce the red-running incentive. Given the compute already spent across
this task, further iteration should be scoped explicitly with the user before
continuing.

## Round 3 plan: harden safety structurally, then retest fuel (in progress)

Rather than rely solely on reward shaping to discourage red-running (which C5
proved can be reintroduced by an unrelated reward change), this round makes
red-running **structurally impossible** via action masking, then re-attempts a
genuine fuel improvement now that the gap term is confirmed safe at its
reverted (`w_gap=0.12`, clip `+30`) values.

- **A1 (safety, mandatory):** in `EcoDrivingEnv.step()`, before applying the
  commanded acceleration, check whether even a worst-case this-step action
  (full `a_max`) would still leave enough room to brake to a stop before the
  line, given the signal is red/yellow. If not, override `a_cmd = a_min` for
  that step regardless of what the policy commanded, and flag `info["forced_brake"]`.
- **A2 (backstop):** raise `r_violation` from `-200` to `-1000`. The mask
  should make this almost never trigger during training; it remains as a
  belt-and-braces deterrent.
- **A3 (verify no deadlock):** 100-seed random-policy stress test confirming
  the mask never lets a red-run through and does not itself cause episodes to
  fail to arrive (target: ≥95% arrival under fully random actions).
- **B1 (the actual research question):** raise `w_fuel` from `1.0` to `1.5`
  (holding the reverted-safe `w_gap`/clip from the ablation) and retrain seed 0
  only, to test whether this closes the +9.8% fuel gap found above — this time
  under a reward/environment where red-running is not just discouraged but
  physically prevented, so any fuel result is trustworthy.

Acceptance for this round: zero red-runs/collisions (all 10 scenarios), ≥9/10
arrivals, paired fuel delta ≤ 0% (safe-comparable subset only), travel time
within +10s of baseline, stop-steps ≤ 20, and confirmation `evaluate.py` used
`best_model.zip`. Results appended below once training completes.

## Round 3 results: red-running eliminated, but a NEW collision regression appears

Trained seed 0 with A1 (action mask) + A2 (`r_violation=-1000`) + B1
(`w_fuel=1.5`), holding the safe `w_gap=0.12`/clip `+30` from the round-2
ablation. Full 400k steps (had to restart once after a session interruption
killed it at 340k with no saved checkpoint). Evaluated with
`eco_driving/scripts/eval_round3.py` (paired fuel delta computed only over
scenarios where both baseline and policy arrive legally).

**A1 worked exactly as designed: 0/10 red-runs.** The `run_action_mask_safety_check`
sanity test (100 random-policy seeds) also confirms this structurally: 0
red-runs with the mask vs. 15 without it, and — critically — 0 timeouts in
both conditions, ruling out mask-induced deadlock. (The mask does not reduce
the orthogonal high collision rate under *pure random* actions — 67-74/100 in
both conditions — which is a pre-existing property of the delta-a action
space's correlated random-walk exploration rear-ending the leader, unrelated
to the mask; `sanity_check.py` now documents and tests for this precisely.)

**But a new collision regression appeared in the trained policy: 4/10
scenarios end in a collision with the leader** (scenarios 500, 505, 506, 507 —
all within 9-16.5s, i.e. very early in the episode). Traced scenario 500
directly: the ego accelerates past the leader's speed (reaching 16 m/s vs. the
leader's ~11-13 m/s) while the gap steadily closes from ~20m to 0 over 14
seconds, with only a weak, far-too-late braking attempt (`a_cmd` only reaches
-0.36 as the gap hits ~1m) — the policy never seriously reacts to the closing
gap until it's already lost. This is a car-following failure, structurally
unrelated to the signal mask (A1 only ever triggers near the stop line).

This is a genuinely new failure mode: the round-2 ablation checkpoint (same
`w_gap`/clip, no mask, `w_fuel=1.0`, `r_violation=-200`) had **zero**
collisions across the same 10 scenarios. The two candidate differences this
round are the mask (A1) and the raised fuel weight (B1); the mask only
intervenes near the stop line so cannot itself explain a collision with the
leader elsewhere on the route — the more likely cause is B1: raising `w_fuel`
50% (1.0→1.5) increased the relative reward for using less fuel, and since
braking/re-accelerating for the leader costs fuel, this likely diluted the
relative importance of the (unchanged in absolute terms) gap-following penalty
enough to make tailgate-and-hope-for-the-best a locally-reinforced behavior for
this seed. Not yet confirmed via ablation (would need a run with the mask kept
but `w_fuel` reverted to 1.0, isolating B1 specifically).

### Round 3 acceptance criteria: 3 of 6 fail

| criterion | result |
|---|---|
| 1. Zero red-runs/collisions | **FAIL** — 0 red-runs (✓) but 4 collisions |
| 2. ≥9/10 arrivals | **FAIL** — 6/10 |
| 3. Paired fuel delta ≤0% (legal-both-sides, n=6) | **FAIL** — +9.0% ± 10.8% |
| 4. Travel time within +10s of baseline | PASS — 59.5s vs 93.2s (biased low: includes 4 short collision episodes) |
| 5. Stop-steps ≤20 | PASS — 12.8 |
| 6. Confirmed `best_model.zip` loaded | PASS |

Per this round's own instructions ("do not silently tune further... wait for
input, do not chain another retrain"), stopping here to report rather than
attempting a further isolation run unprompted. `models/sac_seed0_wfuel15/` and
`models/sac_seed0_ablation_noC5/` are both kept on disk for comparison; neither
is a passing result yet — the former is fuel-competitive-ish but crashes into
the leader, the latter is fully safe but loses on fuel (+9.8%).
