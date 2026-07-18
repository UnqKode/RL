# Round 4–6 Final Report: 5-Seed Aggregate Evaluation

Generated from `models/sac_seed{0,1,2,3,4}_round4/best_model.zip`, evaluated on
the 10 fixed scenario seeds (500–509) against the IDM baseline, via
`eco_driving/scripts/eval_round5.py`. Raw data: `summary_metrics.csv`,
`paired_fuel_delta.csv`, `confound_check.csv` in this directory. Seeds 0–2 are
from round 5; seeds 3–4 were added in round 6 (Task 3) under the byte-identical
configuration, with no exclusions or reseeding.

## Confound verdict (Task 2)

The A1 (signal) and A4 (leader-collision) safety guards are environment-level
and intervene on *any* controller, including the baseline. Measured directly:
running the identical baseline logic on the same 10 scenarios with guards on
vs. off, the guards fire on the baseline 69 times across 10 scenarios and
raise its mean fuel use by **+3.67%** (ratio 1.0367). This exceeds the ±1%
tolerance, so **the confound is confirmed**.

**Decision (per the pre-committed rule): the primary comparison is guarded
policy vs. unguarded baseline.** The unguarded baseline is the legitimate
yardstick because its collision-free safety was independently established
without any guard across 200+ seeds in rounds 1–3; the policy remains
evaluated guarded because its own safety currently depends on the guards
being present. This lowers the round-4 seed-0 headline from the originally
reported −9.6% to **−6.6%** — same direction, smaller magnitude.

## Per-seed results (all 5 seeds)

| seed | arrived | red-runs | collisions | guard-rate | paired fuel Δ (mean ± std, n=10) | mean travel time | mean stop-steps | mean max\|jerk\| |
|---|---|---|---|---|---|---|---|---|
| 0 | 10/10 | 0 | 0 | 0.94% | −6.6% ± 10.9% | 93.85 s | 4.30 | 4.75 |
| 1 | 10/10 | 0 | 0 | 1.08% | −12.6% ± 10.3% | 96.25 s | 2.80 | 5.45 |
| 2 | 10/10 | 0 | 0 | 1.39% | −10.2% ± 11.7% | 94.30 s | 3.90 | 6.06 |
| 3 | 10/10 | 0 | 0 | 1.21% | −8.7% ± 14.0% | 95.15 s | 3.70 | 6.40 |
| 4 | 10/10 | 0 | 0 | 1.62% | −8.8% ± 17.2% | 95.70 s | 6.50 | 6.56 |

**Baseline (unguarded), for reference:** travel time 93.25 s, stop-steps 0.60,
max|jerk| 3.38.

All five seeds independently satisfy every round-4/5/6 acceptance criterion:
zero collisions and zero red-runs (**50/50 scenario-runs across all five
seeds are safe** — structural, as designed), ≥9/10 arrivals (10/10 for all
five), guard-activation rate ≤2% (0.94–1.62%, i.e. the policies are not
riding the safety net), and travel time within +10s of baseline for every
seed (all five within ~3.5s). Seeds 3 and 4 were trained identically to
seeds 0–2 (Round 6, Task 3) — same guards, reward, hyperparameters, 400k
steps — and both converged cleanly on the first attempt with no exclusions.

## Aggregate: 3-seed (round 5) vs. 5-seed (round 6), side by side

Per-seed mean paired fuel deltas (5-seed set): **[−6.57%, −12.56%, −10.21%,
−8.67%, −8.80%]**.

| | 3-seed (round 5) | 5-seed (round 6) |
|---|---|---|
| mean | −9.78% | **−9.36%** |
| std (across seeds) | 3.02% | 2.21% |
| SE | 1.74% | 0.99% |
| df | 2 | 4 |
| t-statistic | −5.61 | **−9.46** |
| p-value | 0.030 | **0.0007** |
| 95% CI | [−17.28%, −2.28%] | **[−12.11%, −6.62%]** |

Adding seeds 3 and 4 **strengthened** the result on every axis: the point
estimate barely moved (−9.78% → −9.36%), but the across-seed variance
shrank (std 3.02%→2.21%), the confidence interval narrowed by roughly 40%
(width 15.0%→5.5%), and the p-value dropped by more than an order of
magnitude (0.030→0.0007). Per the round's rules, seeds 3 and 4 entered the
aggregate exactly as measured — no exclusions, no reseeding — and both
happened to land fuel-negative, consistent with the first three.

## 3-seed aggregate (superseded by the 5-seed aggregate above, kept for continuity)

Per-seed mean paired fuel deltas: **[−6.57%, −12.56%, −10.21%]**.

- **Aggregate mean ± std (across seeds): −9.78% ± 3.02%**
- Standard error (n=3): 1.74%
- One-sample t-test against 0 (n=3, df=2): **t = −5.61, p = 0.030**
- 95% CI on the mean: **[−17.28%, −2.28%]**

With only 3 seeds (df=2), this p-value and CI should be read as indicative,
not as strong statistical confirmation — a df=2 t-test is sensitive to the
exact seed values and the CI is wide (spanning roughly −17% to −2%). What can
be said honestly: all three independently-trained seeds landed on the
fuel-negative side with no cherry-picking (all three converged safely on the
first attempt under this round's configuration, unlike every previous round),
and the interval does not include or come close to crossing into
fuel-positive territory, which is a meaningfully more robust result than a
single-seed point estimate.

**One caveat not part of this round's acceptance criteria but worth
disclosing:** mean max|jerk| for the policy (4.75–6.56 across all 5 seeds) is
higher than the unguarded baseline's (3.38). The policy is not smoother than
the baseline in this round's evaluation — the fuel improvement is not paired with
a comfort improvement. This wasn't a round-4/5 pass/fail criterion, but it
would be relevant to any claim that the policy is unambiguously better across
all three original objectives (fuel, time, comfort).

## Round 6, Task 1: jerk decomposition — the caveat is partly, not fully, a guard artifact

Hypothesis: a single guard override forces `a_min=-4`, producing a one-step
jerk spike that a single `max|jerk|` statistic is especially sensitive to.
Re-rolled all 10 scenarios per seed, pooling every step, and computed
max|jerk| with guard-firing steps (and the step immediately after, since the
recovery transition is also elevated) excluded, plus the 95th-percentile
|jerk| over all steps. Full data: `results/round4/jerk_decomposition.csv`.

| driver | max\|jerk\| (all steps) | max\|jerk\| (guard-excluded) | p95 \|jerk\| | steps excluded |
|---|---|---|---|---|
| baseline (unguarded) | 4.23 | 4.23 | 0.91 | 0/1865 |
| seed 0 | 8.07 | 4.63 | 1.45 | 36/1877 |
| seed 1 | 11.25 | 4.72 | 1.34 | 39/1925 |
| seed 2 | 10.34 | 4.76 | 1.75 | 52/1886 |

**The hypothesis holds for the max statistic.** Guard-excluded max|jerk| is
1.09–1.12× the baseline's — within the pre-committed ±15% parity band — using
the decision rule from this round's task: *"comfort is at parity during
normal driving; elevated maxima are confined to rare safety interventions
(~1.9–2.8% of steps, matching the guard-activation rates already reported)."*

**But the 95th-percentile tells a different, less flattering story, and is
reported here rather than left out because the max-statistic passed.**
p95|jerk| is 1.47–1.92× the baseline's *even excluding nothing* (this
percentile is computed over all steps, guard or not, and 95% of ~1900 steps is
already well outside the ~40-step guard-affected tail). This means the
policy's *typical* driving — not just its single worst moment — involves
meaningfully more frequent moderate-to-high jerk events than the baseline's.
The honest combined verdict: the single worst moment of a policy episode is
about as sharp as the baseline's, but a policy episode has a rougher ride
overall than the max-statistic-parity finding alone would suggest.

## Round 6, Task 2: convergence check

Read `evaluations.npz` for each seed: timestep of the best eval checkpoint,
and the slope of the eval-reward curve over the final 100k training steps
(simple linear fit). Full data: `results/round4/convergence_check.csv`.

| seed | best checkpoint (steps) | best eval reward | final-100k slope (per 1k steps) | verdict |
|---|---|---|---|---|
| 0 | 380,000 | −19.49 | +0.0376 | **still-improving** (best ≥350k, slope>0) |
| 1 | 300,000 | −5.20 | −0.0143 | **converged** (best <350k, slope≤0) |
| 2 | 180,000 | −15.29 | +0.3516 | **ambiguous** — see below |

Seed 2 doesn't cleanly fit either bucket as strictly defined: its best
checkpoint came early (180k, satisfying the "converged" timing criterion) but
its final-100k slope is clearly positive and about 10× steeper than seed 0's
(+0.35 vs. +0.04 per 1k steps) — the opposite of what "converged" requires.
This reads as a noisy, non-monotonic reward curve rather than a clean
convergence, but it does not meet the *conjunctive* still-improving
definition (`best ≥350k AND slope>0`) either, since its best checkpoint is
early. Per the round's strict gating rule, **only seed 0 is gated into Task
4** (extended training); seed 2's anomaly is reported here as a caveat but not
acted on, since the task's rule is explicit about not training longer "just
in case."

## What this evidence can and cannot support

This evaluation is on a **single simulated intersection** with one fixed-time
signal cycle, a stop-and-go leader model, and fuel constants that are
representative/illustrative rather than calibrated to any real vehicle (see
the project README). Ten fixed evaluation scenarios and five training seeds
provide meaningfully more evidence than a single run, but they do not
constitute a large-sample or multi-intersection validation. What the evidence
supports: under this specific simulated environment, a SAC policy trained with
structural safety guards (A1/A4) and a fuel-forward reward weighting
(`w_fuel=1.5`) can reduce fuel consumption by roughly 6.6–12.6% (per-seed
means) relative to a conventional IDM-following baseline, without any observed
safety violations across 50 seed-scenario combinations, while remaining
smoother-comfort is not established — guard-excluded max|jerk| is at parity
with baseline but p95|jerk| is not (Round 6, Task 1). It does not
demonstrate this holds at other intersections, other traffic patterns, other
fuel models, or with real vehicle actuation/latency, and the guard-on-baseline
confound finding is a reminder that any structural safety mechanism added to
an environment should be checked against *all* controllers evaluated in it,
not just the one being trained.
