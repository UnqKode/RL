"""Round-5: commit round-4 evaluation evidence, run the guard-on-baseline
confound check, and (once trained) aggregate seeds 0/1/2 under the round-4
configuration. See CHANGES.md ("Round 5") for context.

Usage:
    python -m eco_driving.scripts.eval_round5 --seeds 0            # seed0 only (Task 1+2)
    python -m eco_driving.scripts.eval_round5 --seeds 0 1 2        # full aggregate (Task 3+4)
"""
import argparse
import csv
import os

import numpy as np

from ..config import EnvConfig
from ..baseline.idm_driver import IDMBaselineDriver
from .evaluate import SCENARIO_SEEDS, load_policy, rollout_policy, rollout_baseline, summarize_metric

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results", "round4")


def model_dir_for(seed):
    return os.path.join(MODELS_DIR, f"sac_seed{seed}_round4")


# ---------------------------------------------------------------------------
# Task 2: guard-on-baseline confound check
# ---------------------------------------------------------------------------
def run_confound_check():
    cfg_guarded = EnvConfig()  # mask_enabled=True (round-4 environment)
    cfg_unguarded = EnvConfig(mask_enabled=False)  # round-3 environment (no guards)
    driver = IDMBaselineDriver(cfg_guarded)  # same IDM logic/params either way

    rows = []
    for s in SCENARIO_SEEDS:
        _, guarded_summary = rollout_baseline(cfg_guarded, s, driver)
        _, unguarded_summary = rollout_baseline(cfg_unguarded, s, driver)
        ratio = guarded_summary["total_fuel_mL"] / unguarded_summary["total_fuel_mL"]
        rows.append(dict(
            scenario_seed=s,
            guarded_fuel_mL=guarded_summary["total_fuel_mL"],
            unguarded_fuel_mL=unguarded_summary["total_fuel_mL"],
            fuel_ratio=ratio,
            guarded_guard_a1=guarded_summary["guard_a1_count"],
            guarded_guard_a4=guarded_summary["guard_a4_count"],
            guarded_guard_rate=guarded_summary["guard_rate"],
            guarded_jerk=guarded_summary["max_abs_jerk"],
            unguarded_jerk=unguarded_summary["max_abs_jerk"],
        ))

    print("\n" + "=" * 90)
    print("TASK 2: GUARD-ON-BASELINE CONFOUND CHECK")
    print("=" * 90)
    print(f"{'seed':>6} {'guard_a1':>9} {'guard_a4':>9} {'guard_rate':>11} "
          f"{'fuel_guarded':>13} {'fuel_unguarded':>15} {'ratio':>8} {'jerk_g':>7} {'jerk_u':>7}")
    for r in rows:
        print(f"{r['scenario_seed']:>6} {r['guarded_guard_a1']:>9} {r['guarded_guard_a4']:>9} "
              f"{r['guarded_guard_rate']*100:>10.2f}% {r['guarded_fuel_mL']:>13.2f} "
              f"{r['unguarded_fuel_mL']:>15.2f} {r['fuel_ratio']:>8.3f} "
              f"{r['guarded_jerk']:>7.2f} {r['unguarded_jerk']:>7.2f}")

    total_guard_events = sum(r["guarded_guard_a1"] + r["guarded_guard_a4"] for r in rows)
    mean_ratio = float(np.mean([r["fuel_ratio"] for r in rows]))
    mean_pct_change = (mean_ratio - 1.0) * 100
    print(f"\nTotal guard activations on baseline across 10 scenarios: {total_guard_events}")
    print(f"Mean baseline fuel ratio (guarded/unguarded): {mean_ratio:.4f}  "
          f"({mean_pct_change:+.2f}% change)")

    if abs(mean_pct_change) <= 1.0:
        verdict = "CONFOUND RULED OUT (baseline fuel change within +/-1%)"
        primary = "guarded_vs_guarded"
    else:
        verdict = "CONFOUND CONFIRMED (baseline fuel changed >1% under guards)"
        primary = "guarded_policy_vs_unguarded_baseline"
    print(f"\nDECISION: {verdict}")
    print(f"Primary comparison to use: {primary}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "confound_check.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved confound-check table to {path}")
    return rows, mean_pct_change, primary


# ---------------------------------------------------------------------------
# Task 1/3: per-seed evaluation producing results/round4/ CSVs
# ---------------------------------------------------------------------------
def evaluate_seed(seed, cfg_guarded, cfg_unguarded, driver_guarded, driver_unguarded):
    model_dir = model_dir_for(seed)
    model, vecnorm = load_policy(model_dir)

    rows_summary = []
    rows_paired = []
    for s in SCENARIO_SEEDS:
        _, base_g = rollout_baseline(cfg_guarded, s, driver_guarded)
        _, base_u = rollout_baseline(cfg_unguarded, s, driver_unguarded)
        _, pol = rollout_policy(cfg_guarded, s, model, vecnorm)  # policy always guarded (safety depends on it)

        for driver_name, summ in [("baseline_guarded", base_g), ("baseline_unguarded", base_u),
                                   ("sac_policy", pol)]:
            row = dict(summ)
            row["driver"] = driver_name
            row["policy_seed"] = seed
            rows_summary.append(row)

        base_g_legal = base_g["arrived"] and not base_g["red_run"] and not base_g["collision"]
        base_u_legal = base_u["arrived"] and not base_u["red_run"] and not base_u["collision"]
        pol_legal = pol["arrived"] and not pol["red_run"] and not pol["collision"]

        def delta_pct(fuel_sac, fuel_base):
            return (fuel_sac - fuel_base) / fuel_base * 100

        rows_paired.append(dict(
            policy_seed=seed, scenario_seed=s,
            base_arrived=base_g["arrived"], base_red_run=base_g["red_run"], base_collision=base_g["collision"],
            sac_arrived=pol["arrived"], sac_red_run=pol["red_run"], sac_collision=pol["collision"],
            both_arrived_guarded=base_g_legal and pol_legal,
            fuel_base_guarded_mL=base_g["total_fuel_mL"], fuel_base_unguarded_mL=base_u["total_fuel_mL"],
            fuel_sac_mL=pol["total_fuel_mL"],
            fuel_delta_pct_vs_guarded_baseline=delta_pct(pol["total_fuel_mL"], base_g["total_fuel_mL"])
            if base_g_legal and pol_legal else None,
            fuel_delta_pct_vs_unguarded_baseline=delta_pct(pol["total_fuel_mL"], base_u["total_fuel_mL"])
            if base_u_legal and pol_legal else None,
            both_arrived_unguarded_baseline=base_u_legal and pol_legal,
        ))
    return rows_summary, rows_paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    confound_rows, mean_pct_change, primary = run_confound_check()

    cfg_guarded = EnvConfig()
    cfg_unguarded = EnvConfig(mask_enabled=False)
    driver_guarded = IDMBaselineDriver(cfg_guarded)
    driver_unguarded = IDMBaselineDriver(cfg_unguarded)

    all_summary_rows = []
    all_paired_rows = []
    for seed in args.seeds:
        print(f"\n{'='*90}\nEVALUATING seed {seed}\n{'='*90}")
        rows_summary, rows_paired = evaluate_seed(seed, cfg_guarded, cfg_unguarded,
                                                    driver_guarded, driver_unguarded)
        all_summary_rows.extend(rows_summary)
        all_paired_rows.extend(rows_paired)

        pol_rows = [r for r in rows_summary if r["driver"] == "sac_policy"]
        n_arrived = sum(r["arrived"] for r in pol_rows)
        n_redrun = sum(r["red_run"] for r in pol_rows)
        n_collision = sum(r["collision"] for r in pol_rows)
        guard_rate = np.mean([r["guard_rate"] for r in pol_rows])
        print(f"seed {seed}: arrived {n_arrived}/10  red_run {n_redrun}  collision {n_collision}  "
              f"guard_rate {guard_rate*100:.2f}%")

        deltas_primary_key = "fuel_delta_pct_vs_guarded_baseline" if primary == "guarded_vs_guarded" \
            else "fuel_delta_pct_vs_unguarded_baseline"
        deltas = [r[deltas_primary_key] for r in rows_paired if r[deltas_primary_key] is not None]
        print(f"seed {seed}: paired fuel delta ({primary}): "
              f"{np.mean(deltas):+.1f}% +/- {np.std(deltas):.1f}%  (n={len(deltas)})")

    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = list(all_summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary_rows)
    print(f"\nSaved {summary_path}")

    paired_path = os.path.join(RESULTS_DIR, "paired_fuel_delta.csv")
    with open(paired_path, "w", newline="") as f:
        fieldnames = list(all_paired_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_paired_rows)
    print(f"Saved {paired_path}")


if __name__ == "__main__":
    main()
