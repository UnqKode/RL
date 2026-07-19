"""Round 7, Task 3: baseline runs on the pre-registered 30-scenario set
(seeds 500-529) -- unguarded (primary yardstick) and guarded (confound
re-measurement on the expanded set). Needs no trained model.
"""
import csv
import os

import numpy as np

from ..config import EnvConfig
from ..baseline.idm_driver import IDMBaselineDriver
from .evaluate import rollout_baseline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results", "round7")

SCENARIOS_30 = list(range(500, 530))  # pre-registered, fixed


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_guarded = EnvConfig()
    cfg_unguarded = EnvConfig(mask_enabled=False)
    driver_guarded = IDMBaselineDriver(cfg_guarded)
    driver_unguarded = IDMBaselineDriver(cfg_unguarded)

    baseline_rows = []
    confound_rows = []
    for s in SCENARIOS_30:
        _, summ_u = rollout_baseline(cfg_unguarded, s, driver_unguarded)
        _, summ_g = rollout_baseline(cfg_guarded, s, driver_guarded)

        row_u = dict(summ_u); row_u["driver"] = "baseline_unguarded"
        row_g = dict(summ_g); row_g["driver"] = "baseline_guarded"
        baseline_rows.append(row_u)
        baseline_rows.append(row_g)

        ratio = summ_g["total_fuel_mL"] / summ_u["total_fuel_mL"]
        confound_rows.append(dict(
            scenario_seed=s,
            guarded_fuel_mL=summ_g["total_fuel_mL"], unguarded_fuel_mL=summ_u["total_fuel_mL"],
            fuel_ratio=ratio,
            guarded_guard_a1=summ_g["guard_a1_count"], guarded_guard_a4=summ_g["guard_a4_count"],
            guarded_guard_rate=summ_g["guard_rate"],
            guarded_jerk=summ_g["max_abs_jerk"], unguarded_jerk=summ_u["max_abs_jerk"],
        ))
        print(f"scenario {s}: unguarded[arrived={summ_u['arrived']} fuel={summ_u['total_fuel_mL']:.1f}] "
              f"guarded[arrived={summ_g['arrived']} fuel={summ_g['total_fuel_mL']:.1f} "
              f"guard_a1={summ_g['guard_a1_count']} guard_a4={summ_g['guard_a4_count']}]")

    baseline_path = os.path.join(RESULTS_DIR, "baseline_30.csv")
    with open(baseline_path, "w", newline="") as f:
        fieldnames = list(baseline_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(baseline_rows)
    print(f"\nSaved {baseline_path}")

    confound_path = os.path.join(RESULTS_DIR, "confound_check_30.csv")
    with open(confound_path, "w", newline="") as f:
        fieldnames = list(confound_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(confound_rows)
    print(f"Saved {confound_path}")

    n_arrived_u = sum(r["arrived"] for r in baseline_rows if r["driver"] == "baseline_unguarded")
    n_redrun_u = sum(r["red_run"] for r in baseline_rows if r["driver"] == "baseline_unguarded")
    n_collision_u = sum(r["collision"] for r in baseline_rows if r["driver"] == "baseline_unguarded")
    total_guard_events = sum(r["guarded_guard_a1"] + r["guarded_guard_a4"] for r in confound_rows)
    mean_ratio = float(np.mean([r["fuel_ratio"] for r in confound_rows]))
    print(f"\nUnguarded baseline (30 scenarios): arrived {n_arrived_u}/30  red_run {n_redrun_u}  "
          f"collision {n_collision_u}")
    print(f"Guard activations on baseline (30 scenarios): {total_guard_events}")
    print(f"Mean baseline fuel ratio (guarded/unguarded): {mean_ratio:.4f}  "
          f"({(mean_ratio-1)*100:+.2f}% change)")


if __name__ == "__main__":
    main()
