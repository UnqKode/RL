# Round 7 Pre-Registration

Committed before any Round 7 model exists or any evaluation is run. This
document fixes the protocol; it is not modified after training/evaluation
begins except to disclose a seed failure per the rules below.

## Evaluation scenario set

**Fixed: scenario seeds 500–529 inclusive (30 scenarios).** No additions or
removals for any reason, at any point in this round.

## Training seeds

**Fixed: seeds 0–6 inclusive (7 seeds), fixed before launch.** A seed that
fails training twice (crash or stall, restarted once with the same seed
number per the round's rules) is dropped from the aggregate with explicit
disclosure — degrees of freedom shrink accordingly. No seed is ever
substituted for a different seed number.

## Primary comparison

**Guarded policy vs. UNGUARDED baseline**, per the round-5 confound rule
(the A1/A4 safety guards intervene on any controller including the baseline;
the unguarded baseline is the legitimate yardstick because its safety was
independently established without any guard across 200+ seeds in rounds 1–3).
Metric: per-scenario paired fuel delta `(fuel_sac - fuel_base_unguarded) /
fuel_base_unguarded`, restricted to scenarios where both sides arrive legally
(no red-run, no collision).

Two-level aggregation: per-scenario deltas → per-seed mean (over the 30
scenarios) → across-seed mean (over the 7 seeds), with the standard error,
one-sample t-test (df=6), and 95% CI computed on the 7 per-seed means.

## Primary endpoint

**The 7-seed mean paired fuel delta with its 95% CI.**

## Secondary endpoints

- Travel time delta (policy vs. unguarded baseline).
- Stop-steps (policy).
- Guard-activation rate (policy), per seed.
- Jerk decomposition: max|jerk| (all steps), max|jerk| (guard-firing step and
  the recovery step after excluded), and p95|jerk| (all steps) — per seed and
  for the unguarded baseline, following the round-6 Task 1 method exactly.

## Decision rule

Whatever the 1M-step × 30-scenario protocol produces **is the reported
primary result**, including if it is weaker than the round-6 (400k-step,
10-scenario) result. The 400k×10 aggregate is reported alongside for
continuity, clearly labeled **historical**, not primary. No cherry-picking
between the two after the fact — the 1M×30 protocol is primary because it is
pre-registered here as such, independent of which number looks better.

## Configuration (Task 1, applied after this commit)

Byte-identical to the round-4/5/6 configuration except:
- `total_timesteps`: 400,000 → **1,000,000**
- Linear LR-decay horizon rescaled to the new total: still 3e-4 → 1e-4, now
  spread over 1,000,000 steps instead of 400,000 (same endpoints, same
  schedule shape, just stretched).

Everything else unchanged: guards A1/A4 (structural, forward-simulating
stopping-distance checks), `w_guard=0.5`, `w_fuel=1.5`, `w_gap=0.12`/clip
`+30`, `r_violation=-1000`, buffer size, batch size, `target_entropy=-1.5`,
`EvalCallback` every 10k steps with best-model saving.

## What happens if a seed fails

Per this round's rules: restart the failed seed once with the same seed
number. If it fails a second time, drop it from the aggregate, disclose the
failure explicitly in the final report, and proceed with the remaining seeds
(df = n−1 where n is the number of seeds that completed). No substitution of
a different seed number under any circumstance.
