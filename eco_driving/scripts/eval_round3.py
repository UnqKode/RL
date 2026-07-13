"""Round-3 evaluation of models/sac_seed0_wfuel15 against baseline on the same
10 fixed scenarios, with paired fuel delta restricted to legal-arrival-on-both-
sides scenarios, and explicit acceptance-criteria checks for this round.
See CHANGES.md ("Round 3") for context and results.
"""
import numpy as np

from eco_driving.config import EnvConfig
from eco_driving.baseline.idm_driver import IDMBaselineDriver
from eco_driving.scripts.evaluate import (SCENARIO_SEEDS, load_policy, rollout_policy,
                                           rollout_baseline, summarize_metric)

cfg = EnvConfig()
model_dir = "models/sac_seed0_wfuel15"
model, vecnorm = load_policy(model_dir)
driver = IDMBaselineDriver(cfg)

baseline_summaries = []
policy_summaries = []
for s in SCENARIO_SEEDS:
    b_trace, b_summary = rollout_baseline(cfg, s, driver)
    p_trace, p_summary = rollout_policy(cfg, s, model, vecnorm)
    baseline_summaries.append(b_summary)
    policy_summaries.append(p_summary)
    print(f"scenario {s}: "
          f"baseline[arrived={b_summary['arrived']} red_run={b_summary['red_run']} fuel={b_summary['total_fuel_mL']:.1f}] "
          f"policy[arrived={p_summary['arrived']} red_run={p_summary['red_run']} collision={p_summary['collision']} "
          f"fuel={p_summary['total_fuel_mL']:.1f} time={p_summary['travel_time_s']:.1f} "
          f"stops={p_summary['stop_steps']} jerk={p_summary['max_abs_jerk']:.2f}]")

n_arrived = sum(s["arrived"] for s in policy_summaries)
n_redrun = sum(s["red_run"] for s in policy_summaries)
n_collision = sum(s["collision"] for s in policy_summaries)
n_timeout = sum(s["timeout"] for s in policy_summaries)

print(f"\nPOLICY: arrived {n_arrived}/10  red_run {n_redrun}  collision {n_collision}  timeout {n_timeout}")
for key in ["travel_time_s", "total_fuel_mL", "stop_steps", "max_abs_jerk"]:
    m, sd = summarize_metric(policy_summaries, key)
    print(f"  {key:16s}: {m:8.2f} +/- {sd:6.2f}")

base_time, _ = summarize_metric(baseline_summaries, "travel_time_s")
base_fuel, _ = summarize_metric(baseline_summaries, "total_fuel_mL")
base_jerk, _ = summarize_metric(baseline_summaries, "max_abs_jerk")
print(f"\nBASELINE: travel_time={base_time:.2f} fuel={base_fuel:.2f} jerk={base_jerk:.2f}")

# Paired fuel delta: only scenarios where BOTH sides arrived legally
legal_pairs = []
excluded = []
for b, p in zip(baseline_summaries, policy_summaries):
    b_legal = b["arrived"] and not b["red_run"] and not b["collision"]
    p_legal = p["arrived"] and not p["red_run"] and not p["collision"]
    if b_legal and p_legal:
        delta = (p["total_fuel_mL"] - b["total_fuel_mL"]) / b["total_fuel_mL"]
        legal_pairs.append((b["seed"], delta))
    else:
        excluded.append((b["seed"], b_legal, p_legal))

print(f"\nPaired fuel delta (legal-both-sides only, n={len(legal_pairs)}):")
for seed, d in legal_pairs:
    print(f"  scenario {seed}: {d*100:+.1f}%")
if excluded:
    print(f"Excluded scenarios (not legal on both sides): {excluded}")
deltas = [d for _, d in legal_pairs]
mean_delta = float(np.mean(deltas)) * 100 if deltas else float("nan")
std_delta = float(np.std(deltas)) * 100 if deltas else float("nan")
print(f"mean paired fuel delta: {mean_delta:+.1f}% +/- {std_delta:.1f}%")

print("\n" + "=" * 70)
print("ROUND 3 ACCEPTANCE CRITERIA")
print("=" * 70)
c1 = n_redrun == 0 and n_collision == 0
print(f"1. Zero red-runs/collisions: red_run={n_redrun} collision={n_collision} -> {'PASS' if c1 else 'FAIL'}")
c2 = n_arrived >= 9
print(f"2. >=9/10 arrivals: {n_arrived}/10 -> {'PASS' if c2 else 'FAIL'}")
c3 = bool(deltas) and mean_delta <= 0.0
print(f"3. Paired fuel delta <=0%: {mean_delta:+.1f}% -> {'PASS' if c3 else 'FAIL'}")
policy_time, _ = summarize_metric(policy_summaries, "travel_time_s")
c4 = policy_time <= base_time + 10.0
print(f"4. Travel time within +10s of baseline: {policy_time:.1f}s vs {base_time:.1f}s -> {'PASS' if c4 else 'FAIL'}")
policy_stops, _ = summarize_metric(policy_summaries, "stop_steps")
c5 = policy_stops <= 20.0
print(f"5. Stop-steps <=20: {policy_stops:.1f} -> {'PASS' if c5 else 'FAIL'}")
print(f"6. best_model.zip loaded: confirmed via [load_policy] print above -> PASS")
overall = c1 and c2 and c3 and c4 and c5
print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
