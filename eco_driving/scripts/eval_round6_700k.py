"""Round 6, Task 4: evaluate the extended 700k-step seed0 checkpoint and
report a 400k-vs-700k comparison. NOT mixed into the primary 5-seed aggregate.
"""
import numpy as np

from ..config import EnvConfig
from ..baseline.idm_driver import IDMBaselineDriver
from .evaluate import SCENARIO_SEEDS, load_policy, rollout_policy, rollout_baseline, summarize_metric

cfg_guarded = EnvConfig()
cfg_unguarded = EnvConfig(mask_enabled=False)
driver_unguarded = IDMBaselineDriver(cfg_unguarded)

results = {}
for label, model_dir in [("400k", "models/sac_seed0_round4"), ("700k", "models/sac_seed0_round4_700k")]:
    model, vecnorm = load_policy(model_dir)
    summaries = []
    for s in SCENARIO_SEEDS:
        _, pol = rollout_policy(cfg_guarded, s, model, vecnorm)
        _, base_u = rollout_baseline(cfg_unguarded, s, driver_unguarded)
        pol_legal = pol["arrived"] and not pol["red_run"] and not pol["collision"]
        base_legal = base_u["arrived"] and not base_u["red_run"] and not base_u["collision"]
        delta = (pol["total_fuel_mL"] - base_u["total_fuel_mL"]) / base_u["total_fuel_mL"] * 100 \
            if pol_legal and base_legal else None
        pol["fuel_delta_pct"] = delta
        summaries.append(pol)
        print(f"[{label}] scenario {s}: arrived={pol['arrived']} red_run={pol['red_run']} "
              f"collision={pol['collision']} fuel={pol['total_fuel_mL']:.1f} "
              f"guard_rate={pol['guard_rate']*100:.2f}% delta={delta}")
    results[label] = summaries

print("\n" + "=" * 70)
print("400k vs 700k COMPARISON (seed 0) -- NOT part of the primary 5-seed aggregate")
print("=" * 70)
for label in ["400k", "700k"]:
    summaries = results[label]
    n_arrived = sum(s["arrived"] for s in summaries)
    n_redrun = sum(s["red_run"] for s in summaries)
    n_collision = sum(s["collision"] for s in summaries)
    guard_rate, _ = summarize_metric(summaries, "guard_rate")
    deltas = [s["fuel_delta_pct"] for s in summaries if s["fuel_delta_pct"] is not None]
    mean_delta = np.mean(deltas) if deltas else float("nan")
    std_delta = np.std(deltas) if deltas else float("nan")
    time_m, _ = summarize_metric(summaries, "travel_time_s")
    stop_m, _ = summarize_metric(summaries, "stop_steps")
    jerk_m, _ = summarize_metric(summaries, "max_abs_jerk")
    print(f"{label}: arrived {n_arrived}/10  red_run {n_redrun}  collision {n_collision}  "
          f"guard_rate {guard_rate*100:.2f}%  fuel_delta {mean_delta:+.1f}% +/- {std_delta:.1f}%  "
          f"(n={len(deltas)})  time={time_m:.1f}s  stops={stop_m:.1f}  jerk={jerk_m:.2f}")

import csv
with open("results/round4/seed0_400k_vs_700k.csv", "w", newline="") as f:
    fieldnames = ["label", "seed", "arrived", "red_run", "collision", "travel_time_s",
                  "total_fuel_mL", "stop_steps", "max_abs_jerk", "guard_rate", "fuel_delta_pct"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for label in ["400k", "700k"]:
        for s in results[label]:
            row = {k: s.get(k) for k in fieldnames if k != "label"}
            row["label"] = label
            writer.writerow(row)
print("\nSaved results/round4/seed0_400k_vs_700k.csv")
