"""Round 6, Task 1: jerk decomposition diagnostic.

For each trained seed's policy and the unguarded baseline, roll out the same
10 fixed scenarios (500-509) and report, pooling all steps across scenarios:
  - max|jerk| including all steps (continuity with prior reports)
  - max|jerk| EXCLUDING steps where a guard fired, and the step immediately
    after (jerk on the recovery step is also elevated by the transition back
    from the forced a_min)
  - 95th-percentile |jerk| over all steps
Evaluation-only: no env/guard/reward changes. Commits to
results/round4/jerk_decomposition.csv.
"""
import csv
import os

import numpy as np

from ..config import EnvConfig
from ..baseline.idm_driver import IDMBaselineDriver
from .evaluate import SCENARIO_SEEDS, load_policy, rollout_policy, rollout_baseline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results", "round4")
MODELS_DIR = os.path.join(ROOT, "models")


def pooled_jerk_stats(traces):
    """traces: list of per-scenario trace dicts (each with 'jerk' and 'guard_fired' lists)."""
    all_jerk = []
    all_excluded_jerk = []
    for trace in traces:
        jerk = trace["jerk"]
        guard = trace["guard_fired"]
        for i, j in enumerate(jerk):
            all_jerk.append(abs(j))
            excluded = guard[i] or (i > 0 and guard[i - 1])
            if not excluded:
                all_excluded_jerk.append(abs(j))
    max_all = float(np.max(all_jerk)) if all_jerk else float("nan")
    max_excl = float(np.max(all_excluded_jerk)) if all_excluded_jerk else float("nan")
    p95_all = float(np.percentile(all_jerk, 95)) if all_jerk else float("nan")
    n_excluded_steps = len(all_jerk) - len(all_excluded_jerk)
    return dict(max_jerk_all=max_all, max_jerk_excl_guard=max_excl, p95_jerk_all=p95_all,
                n_steps=len(all_jerk), n_excluded_steps=n_excluded_steps)


def main():
    cfg_guarded = EnvConfig()
    cfg_unguarded = EnvConfig(mask_enabled=False)
    driver_unguarded = IDMBaselineDriver(cfg_unguarded)

    rows = []

    # Unguarded baseline (the primary-comparison yardstick)
    traces = []
    for s in SCENARIO_SEEDS:
        trace, _ = rollout_baseline(cfg_unguarded, s, driver_unguarded)
        traces.append(trace)
    stats = pooled_jerk_stats(traces)
    stats["driver"] = "baseline_unguarded"
    rows.append(stats)
    print(f"baseline_unguarded: {stats}")

    # Each trained policy seed (guarded, as always evaluated)
    for seed in [0, 1, 2]:
        model_dir = os.path.join(MODELS_DIR, f"sac_seed{seed}_round4")
        model, vecnorm = load_policy(model_dir)
        traces = []
        for s in SCENARIO_SEEDS:
            trace, _ = rollout_policy(cfg_guarded, s, model, vecnorm)
            traces.append(trace)
        stats = pooled_jerk_stats(traces)
        stats["driver"] = f"sac_seed{seed}"
        rows.append(stats)
        print(f"seed {seed}: {stats}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "jerk_decomposition.csv")
    fieldnames = ["driver", "max_jerk_all", "max_jerk_excl_guard", "p95_jerk_all",
                  "n_steps", "n_excluded_steps"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved {path}")

    # Decision language
    base = rows[0]
    print("\n--- Verdict per seed ---")
    for r in rows[1:]:
        excl_ratio = r["max_jerk_excl_guard"] / base["max_jerk_all"]
        p95_ratio = r["p95_jerk_all"] / base["p95_jerk_all"]
        within_15pct = abs(excl_ratio - 1.0) <= 0.15
        verdict = ("comfort at parity during normal driving; elevated maxima confined to rare "
                   "safety interventions" if within_15pct else
                   "caveat stands: guard-excluded jerk still clearly elevated vs baseline")
        print(f"{r['driver']}: max_excl/base_max={excl_ratio:.2f}  p95/base_p95={p95_ratio:.2f}  "
              f"n_excluded_steps={r['n_excluded_steps']}/{r['n_steps']}  -> {verdict}")


if __name__ == "__main__":
    main()
