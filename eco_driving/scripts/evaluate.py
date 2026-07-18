"""Deterministic evaluation of the trained SAC policy vs. the IDM+late-braking
baseline on fixed scenario seeds, plus plots and a learning-curve figure.

Usage:
    python -m eco_driving.scripts.evaluate
"""
import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from ..config import EnvConfig, TrainConfig
from ..envs import EcoDrivingEnv
from ..envs.signal import GREEN, YELLOW, RED
from ..baseline.idm_driver import IDMBaselineDriver

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

SCENARIO_SEEDS = list(range(500, 510))  # fixed, disjoint from training seeds 0/1/2

PHASE_COLOR = {GREEN: "#2ca02c", YELLOW: "#ffbf00", RED: "#d62728"}


# ---------------------------------------------------------------------------
def load_policy(model_dir):
    model_path = os.path.join(model_dir, "best_model.zip")
    print(f"[load_policy] loading {model_path}")
    model = SAC.load(model_path, device="cpu")
    with open(os.path.join(model_dir, "vecnormalize.pkl"), "rb") as f:
        vecnorm = pickle.load(f)
    return model, vecnorm


def policy_action(model, vecnorm, obs):
    norm_obs = vecnorm.normalize_obs(obs.reshape(1, -1)).astype(np.float32)
    action, _ = model.predict(norm_obs, deterministic=True)
    return float(np.asarray(action).reshape(-1)[0])


# ---------------------------------------------------------------------------
def rollout(cfg: EnvConfig, seed: int, act_fn):
    """act_fn(env) -> float acceleration command. Returns (trace: dict of lists, summary: dict)."""
    env = EcoDrivingEnv(cfg)
    obs, info = env.reset(seed=seed)
    trace = defaultdict(list)
    stop_steps = 0
    max_abs_jerk = 0.0
    n_steps = 0
    n_guard_a1 = 0
    n_guard_a4 = 0
    terminated = truncated = False
    info_last = info
    while not (terminated or truncated):
        sig = env.signal.state(env.t)
        gap, rel_v = env.leader.gap_and_relv(env.x, env.v)
        a = act_fn(env, sig, gap, rel_v)
        obs, r, terminated, truncated, info = env.step(np.array([a], dtype=np.float32))
        trace["t"].append(info["t"])
        trace["x"].append(info["x"])
        trace["v"].append(info["v"])
        trace["fuel_mL"].append(info["fuel_mL"])
        trace["jerk"].append(info["jerk"])
        trace["gap"].append(min(gap, cfg.obs_gap_cap))
        trace["desired_gap"].append(cfg.min_gap + cfg.time_headway * info["v"])
        trace["leader_present"].append(env.leader.present)
        trace["sig_phase"].append(info["sig_phase"])
        trace["guard_fired"].append(bool(info.get("forced_brake") or info.get("forced_brake_leader")))
        if info["v"] < 0.3:
            stop_steps += 1
        max_abs_jerk = max(max_abs_jerk, abs(info["jerk"]))
        n_steps += 1
        if info.get("forced_brake"):
            n_guard_a1 += 1
        if info.get("forced_brake_leader"):
            n_guard_a4 += 1
        info_last = info

    trace["cum_fuel_mL"] = list(np.cumsum(trace["fuel_mL"]))
    summary = dict(
        seed=seed,
        arrived=bool(info_last.get("arrived", False)) and terminated and not info_last.get("red_run", False)
        and not info_last.get("collision", False),
        timeout=bool(truncated),
        red_run=bool(info_last.get("red_run", False)),
        collision=bool(info_last.get("collision", False)),
        travel_time_s=env.t,
        total_fuel_mL=sum(trace["fuel_mL"]),
        stop_steps=stop_steps,
        max_abs_jerk=max_abs_jerk,
        leader_present=env.leader.present,
        n_steps=n_steps,
        guard_a1_count=n_guard_a1,
        guard_a4_count=n_guard_a4,
        guard_rate=(n_guard_a1 + n_guard_a4) / max(n_steps, 1),
    )
    return trace, summary


def rollout_policy(cfg, seed, model, vecnorm):
    def act_fn(env, sig, gap, rel_v):
        obs = env._get_obs()
        return policy_action(model, vecnorm, obs)
    return rollout(cfg, seed, act_fn)


def rollout_baseline(cfg, seed, driver: IDMBaselineDriver):
    def act_fn(env, sig, gap, rel_v):
        dist_to_signal = cfg.signal_pos - env.x
        target_a = driver.act(env.v, gap, rel_v, env.leader.present, dist_to_signal,
                               sig.phase, sig.time_to_change)
        # The env's action is Delta-a (see EcoDrivingEnv docstring); the baseline
        # reasons in absolute target acceleration, so convert. Passing an
        # unclipped delta reproduces a_cmd = clip(target_a, a_min, a_max) exactly
        # -- the baseline's behavior is therefore byte-identical to before the
        # action-space reparameterization.
        return target_a - env.a_prev
    return rollout(cfg, seed, act_fn)


# ---------------------------------------------------------------------------
def evaluate_all_seeds(cfg: EnvConfig, seeds, scenario_seeds):
    driver = IDMBaselineDriver(cfg)
    baseline_summaries = []
    baseline_traces = {}
    for s in scenario_seeds:
        trace, summary = rollout_baseline(cfg, s, driver)
        baseline_summaries.append(summary)
        baseline_traces[s] = trace

    policy_summaries_by_seed = {}
    policy_traces_by_seed = {}
    for seed in seeds:
        model_dir = os.path.join(MODELS_DIR, f"sac_seed{seed}")
        model, vecnorm = load_policy(model_dir)
        summaries = []
        traces = {}
        for s in scenario_seeds:
            trace, summary = rollout_policy(cfg, s, model, vecnorm)
            summaries.append(summary)
            traces[s] = trace
        policy_summaries_by_seed[seed] = summaries
        policy_traces_by_seed[seed] = traces

    return baseline_summaries, baseline_traces, policy_summaries_by_seed, policy_traces_by_seed


def summarize_metric(summaries, key):
    vals = [s[key] for s in summaries]
    return float(np.mean(vals)), float(np.std(vals))


def paired_fuel_deltas(baseline_summaries, policy_summaries):
    """(fuel_sac - fuel_base)/fuel_base for scenarios where both baseline and
    policy arrived, matched by scenario order (both iterate SCENARIO_SEEDS)."""
    deltas = []
    for b, p in zip(baseline_summaries, policy_summaries):
        assert b["seed"] == p["seed"]
        if b["arrived"] and p["arrived"]:
            deltas.append((p["total_fuel_mL"] - b["total_fuel_mL"]) / b["total_fuel_mL"])
    return deltas


def print_report(baseline_summaries, policy_summaries_by_seed):
    print("\n" + "=" * 70)
    print("BASELINE (IDM + late braking)")
    print("=" * 70)
    for key in ["travel_time_s", "total_fuel_mL", "stop_steps", "max_abs_jerk"]:
        m, sd = summarize_metric(baseline_summaries, key)
        print(f"  {key:16s}: {m:8.2f} +/- {sd:6.2f}")
    n_arrived = sum(s["arrived"] for s in baseline_summaries)
    n_redrun = sum(s["red_run"] for s in baseline_summaries)
    n_timeout = sum(s["timeout"] for s in baseline_summaries)
    n_collision = sum(s["collision"] for s in baseline_summaries)
    print(f"  arrived: {n_arrived}/{len(baseline_summaries)}  red_run: {n_redrun}  "
          f"timeout: {n_timeout}  collision: {n_collision}")

    print("\n" + "=" * 70)
    print("SAC POLICY (per seed, mean +/- std over scenarios; then mean +/- std over seeds)")
    print("=" * 70)
    per_seed_means = defaultdict(list)
    fuel_delta_by_seed = {}
    for seed, summaries in policy_summaries_by_seed.items():
        n_arrived = sum(s["arrived"] for s in summaries)
        n_redrun = sum(s["red_run"] for s in summaries)
        n_timeout = sum(s["timeout"] for s in summaries)
        n_collision = sum(s["collision"] for s in summaries)
        print(f"\n  -- seed {seed} -- arrived {n_arrived}/{len(summaries)}  red_run {n_redrun}  "
              f"timeout {n_timeout}  collision {n_collision}")
        for key in ["travel_time_s", "total_fuel_mL", "stop_steps", "max_abs_jerk"]:
            m, sd = summarize_metric(summaries, key)
            print(f"     {key:16s}: {m:8.2f} +/- {sd:6.2f}")
            per_seed_means[key].append(m)

        deltas = paired_fuel_deltas(baseline_summaries, summaries)
        fuel_delta_by_seed[seed] = deltas
        mean_delta = float(np.mean(deltas)) * 100 if deltas else float("nan")
        print(f"     paired fuel delta vs baseline: {mean_delta:+.1f}%  (n={len(deltas)} paired arrivals)")

    print("\n  -- across seeds --")
    for key in ["travel_time_s", "total_fuel_mL", "stop_steps", "max_abs_jerk"]:
        vals = per_seed_means[key]
        print(f"     {key:16s}: {np.mean(vals):8.2f} +/- {np.std(vals):6.2f}  (seed means: {['%.2f' % v for v in vals]})")
    seed_mean_deltas = [float(np.mean(d)) * 100 for d in fuel_delta_by_seed.values() if d]
    if seed_mean_deltas:
        print(f"     paired fuel delta   : {np.mean(seed_mean_deltas):+7.1f}% +/- {np.std(seed_mean_deltas):5.1f}%  "
              f"(seed means: {['%+.1f%%' % v for v in seed_mean_deltas]})")

    return fuel_delta_by_seed


def save_csv(baseline_summaries, policy_summaries_by_seed, path):
    import csv
    rows = []
    for s in baseline_summaries:
        row = dict(s); row["driver"] = "baseline"; row["policy_seed"] = ""
        rows.append(row)
    for seed, summaries in policy_summaries_by_seed.items():
        for s in summaries:
            row = dict(s); row["driver"] = "sac_policy"; row["policy_seed"] = seed
            rows.append(row)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-scenario metrics to {path}")


def save_paired_fuel_csv(baseline_summaries, policy_summaries_by_seed, path):
    import csv
    rows = []
    for seed, summaries in policy_summaries_by_seed.items():
        for b, p in zip(baseline_summaries, summaries):
            both_arrived = b["arrived"] and p["arrived"]
            delta = (p["total_fuel_mL"] - b["total_fuel_mL"]) / b["total_fuel_mL"] if both_arrived else None
            rows.append(dict(policy_seed=seed, scenario_seed=b["seed"], both_arrived=both_arrived,
                              fuel_base_mL=b["total_fuel_mL"], fuel_sac_mL=p["total_fuel_mL"],
                              fuel_delta_pct=None if delta is None else delta * 100))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved paired fuel-delta table to {path}")


def check_acceptance_criteria(baseline_summaries, policy_summaries_by_seed, seeds):
    """Print PASS/FAIL for each of the 7 acceptance criteria from the task spec."""
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)
    results = {}

    # 1. Arrival: every seed >= 9/10, zero red-runs, zero collisions.
    arrival_ok = True
    for seed, summaries in policy_summaries_by_seed.items():
        n_arrived = sum(s["arrived"] for s in summaries)
        n_redrun = sum(s["red_run"] for s in summaries)
        n_collision = sum(s["collision"] for s in summaries)
        seed_ok = n_arrived >= 9 and n_redrun == 0 and n_collision == 0
        arrival_ok &= seed_ok
        print(f"  [1. Arrival]     seed {seed}: {n_arrived}/10 arrived, {n_redrun} red-runs, "
              f"{n_collision} collisions -> {'PASS' if seed_ok else 'FAIL'}")
    results["1_arrival"] = arrival_ok

    # 2. Stops: mean stop-steps per seed <= 15.
    stops_ok = True
    for seed, summaries in policy_summaries_by_seed.items():
        m, _ = summarize_metric(summaries, "stop_steps")
        seed_ok = m <= 15.0
        stops_ok &= seed_ok
        print(f"  [2. Stops]       seed {seed}: mean stop-steps={m:.1f} (<=15) -> {'PASS' if seed_ok else 'FAIL'}")
    results["2_stops"] = stops_ok

    # 3. Fuel: paired delta <= -5% for >= 2 of 3 seeds, no seed worse than +2%.
    n_seeds = len(policy_summaries_by_seed)
    n_good = 0
    fuel_ok = True
    for seed, summaries in policy_summaries_by_seed.items():
        deltas = paired_fuel_deltas(baseline_summaries, summaries)
        mean_delta = float(np.mean(deltas)) if deltas else float("nan")
        good = mean_delta <= -0.05
        bad = mean_delta > 0.02
        n_good += int(good)
        if bad:
            fuel_ok = False
        print(f"  [3. Fuel]        seed {seed}: paired delta={mean_delta*100:+.1f}% "
              f"({'<=-5% GOOD' if good else ('>+2% BAD' if bad else 'neutral')})")
    fuel_ok = fuel_ok and (n_good >= 2)
    print(f"  [3. Fuel]        {n_good}/{n_seeds} seeds <=-5%, all seeds <=+2%? "
          f"-> {'PASS' if fuel_ok else 'FAIL'}")
    results["3_fuel"] = fuel_ok

    # 4. Time: mean travel time within +10s of baseline per seed.
    base_time, _ = summarize_metric(baseline_summaries, "travel_time_s")
    time_ok = True
    for seed, summaries in policy_summaries_by_seed.items():
        m, _ = summarize_metric(summaries, "travel_time_s")
        seed_ok = m <= base_time + 10.0
        time_ok &= seed_ok
        print(f"  [4. Time]        seed {seed}: mean time={m:.1f}s vs baseline {base_time:.1f}s "
              f"(+10s budget) -> {'PASS' if seed_ok else 'FAIL'}")
    results["4_time"] = time_ok

    # 5. Smoothness: mean max|jerk| <= baseline's.
    base_jerk, _ = summarize_metric(baseline_summaries, "max_abs_jerk")
    jerk_ok = True
    for seed, summaries in policy_summaries_by_seed.items():
        m, _ = summarize_metric(summaries, "max_abs_jerk")
        seed_ok = m <= base_jerk
        jerk_ok &= seed_ok
        print(f"  [5. Smoothness]  seed {seed}: mean max|jerk|={m:.2f} vs baseline {base_jerk:.2f} "
              f"-> {'PASS' if seed_ok else 'FAIL'}")
    results["5_smoothness"] = jerk_ok

    print(f"\n  [6. Following]   see results/plots/gap_vs_time.png (visual/manual check)")
    print(f"  [7. Stability]   see learning_curves.png + per-seed evaluations.npz "
          f"(checked separately in evaluate_stability())")

    overall = all(results.values())
    print(f"\n  OVERALL (criteria 1-5): {'PASS' if overall else 'FAIL'}")
    return results


def check_training_stability(seeds, model_dir_fn):
    """Criterion 7: last three EvalCallback points within 15% of the best point."""
    print("\n" + "=" * 70)
    print("CRITERION 7: TRAINING STABILITY (last 3 eval points vs best)")
    print("=" * 70)
    all_ok = True
    for seed in seeds:
        npz_path = os.path.join(model_dir_fn(seed), "evaluations.npz")
        if not os.path.exists(npz_path):
            print(f"  seed {seed}: no evaluations.npz found, skipping")
            continue
        data = np.load(npz_path)
        results_arr = data["results"].mean(axis=1)
        best = np.max(results_arr)
        last3 = results_arr[-3:]
        rel_dev = np.abs(last3 - best) / max(abs(best), 1e-6)
        seed_ok = bool(np.all(rel_dev <= 0.15))
        all_ok &= seed_ok
        print(f"  seed {seed}: best={best:.1f}  last3={list(np.round(last3, 1))}  "
              f"rel_dev={list(np.round(rel_dev, 3))} -> {'PASS' if seed_ok else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
def shade_signal_phases(ax, t_arr, phase_arr):
    start = 0
    cur = phase_arr[0]
    for i in range(1, len(phase_arr)):
        if phase_arr[i] != cur:
            ax.axvspan(t_arr[start], t_arr[i], color=PHASE_COLOR.get(cur, "gray"), alpha=0.15, lw=0)
            start = i
            cur = phase_arr[i]
    ax.axvspan(t_arr[start], t_arr[-1], color=PHASE_COLOR.get(cur, "gray"), alpha=0.15, lw=0)


def plot_speed_position(cfg, baseline_trace, policy_trace, seed, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(baseline_trace["x"], baseline_trace["v"], label="baseline (IDM, late braking)", color="tab:red")
    ax.plot(policy_trace["x"], policy_trace["v"], label="SAC eco-policy", color="tab:blue")
    ax.axvline(cfg.signal_pos, color="k", linestyle="--", alpha=0.5, label="signal position")
    ax.set_xlabel("position x [m]")
    ax.set_ylabel("speed v [m/s]")
    ax.set_title(f"Speed vs. position (scenario seed={seed})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_speed_time(cfg, baseline_trace, policy_trace, seed, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    for ax, trace, name in zip(axes, [baseline_trace, policy_trace], ["baseline", "SAC eco-policy"]):
        shade_signal_phases(ax, trace["t"], trace["sig_phase"])
        ax.plot(trace["t"], trace["v"], color="tab:blue" if "SAC" in name else "tab:red")
        ax.set_ylabel("speed v [m/s]")
        ax.set_title(f"{name}: speed vs. time (signal phase shaded: green/yellow/red)")
    axes[-1].set_xlabel("time t [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cumulative_fuel(baseline_trace, policy_trace, seed, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(baseline_trace["t"], baseline_trace["cum_fuel_mL"], label="baseline", color="tab:red")
    ax.plot(policy_trace["t"], policy_trace["cum_fuel_mL"], label="SAC eco-policy", color="tab:blue")
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("cumulative fuel [mL]")
    ax.set_title(f"Cumulative fuel use (scenario seed={seed})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_gap(baseline_trace, policy_trace, seed, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(baseline_trace["t"], baseline_trace["gap"], label="baseline actual gap", color="tab:red")
    ax.plot(baseline_trace["t"], baseline_trace["desired_gap"], label="baseline desired gap",
            color="tab:red", linestyle="--", alpha=0.6)
    ax.plot(policy_trace["t"], policy_trace["gap"], label="policy actual gap", color="tab:blue")
    ax.plot(policy_trace["t"], policy_trace["desired_gap"], label="policy desired gap",
            color="tab:blue", linestyle="--", alpha=0.6)
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("gap to leader [m]")
    ax.set_title(f"Car-following gap vs. time (scenario seed={seed})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_learning_curves(seeds, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    all_ts = []
    curves = []
    for seed in seeds:
        model_dir = os.path.join(MODELS_DIR, f"sac_seed{seed}")
        npz_path = os.path.join(model_dir, "evaluations.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        ts = data["timesteps"]
        results = data["results"].mean(axis=1)
        ax.plot(ts, results, alpha=0.35, color="tab:blue")
        all_ts.append(ts)
        curves.append(results)
    if curves:
        min_len = min(len(c) for c in curves)
        ts_ref = all_ts[0][:min_len]
        stacked = np.stack([c[:min_len] for c in curves])
        mean_c = stacked.mean(axis=0)
        std_c = stacked.std(axis=0)
        ax.plot(ts_ref, mean_c, color="tab:blue", linewidth=2, label="mean across seeds")
        ax.fill_between(ts_ref, mean_c - std_c, mean_c + std_c, color="tab:blue", alpha=0.2)
    ax.set_xlabel("training timesteps")
    ax.set_ylabel("mean eval episode reward")
    ax.set_title("Learning curves (EvalCallback, deterministic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def check_gap_tracking(policy_traces_by_seed, cfg, sustained_thresh_s=10.0):
    """Criterion 6: gap should track the desired headway, not sit in a sustained
    excursion above 2.5x desired while the leader is present -- checked across
    ALL scenario traces (not just the one used for the overlay plot)."""
    print("\n" + "=" * 70)
    print("CRITERION 6: CAR-FOLLOWING (gap tracks desired headway)")
    print("=" * 70)
    all_ok = True
    for seed, traces in policy_traces_by_seed.items():
        max_run_steps = 0
        for trace in traces.values():
            run = 0
            for present, gap, desired in zip(trace["leader_present"], trace["gap"], trace["desired_gap"]):
                if present and gap > 2.5 * desired:
                    run += 1
                    max_run_steps = max(max_run_steps, run)
                else:
                    run = 0
        max_run_s = max_run_steps * cfg.dt
        seed_ok = max_run_s <= sustained_thresh_s
        all_ok &= seed_ok
        print(f"  seed {seed}: longest sustained gap > 2.5x-desired excursion = {max_run_s:.1f}s "
              f"(<= {sustained_thresh_s:.0f}s) -> {'PASS' if seed_ok else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TrainConfig().seeds))
    parser.add_argument("--plot-seed", type=int, default=None, help="which trained seed to use for the overlay plots")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    cfg = EnvConfig()
    baseline_summaries, baseline_traces, policy_summaries_by_seed, policy_traces_by_seed = \
        evaluate_all_seeds(cfg, args.seeds, SCENARIO_SEEDS)

    print_report(baseline_summaries, policy_summaries_by_seed)
    save_csv(baseline_summaries, policy_summaries_by_seed, os.path.join(RESULTS_DIR, "summary_metrics.csv"))
    save_paired_fuel_csv(baseline_summaries, policy_summaries_by_seed,
                         os.path.join(RESULTS_DIR, "paired_fuel_delta.csv"))

    check_acceptance_criteria(baseline_summaries, policy_summaries_by_seed, args.seeds)
    check_gap_tracking(policy_traces_by_seed, cfg)
    check_training_stability(args.seeds, lambda s: os.path.join(MODELS_DIR, f"sac_seed{s}"))

    plot_seed = args.plot_seed or args.seeds[0]
    # pick a scenario with a leader present for the gap plot; else fall back to first
    leader_scenario = next((s for s in SCENARIO_SEEDS if baseline_traces[s]["leader_present"][0]), SCENARIO_SEEDS[0])
    plain_scenario = SCENARIO_SEEDS[0]

    plot_speed_position(cfg, baseline_traces[plain_scenario], policy_traces_by_seed[plot_seed][plain_scenario],
                         plain_scenario, os.path.join(PLOTS_DIR, "speed_vs_position.png"))
    plot_speed_time(cfg, baseline_traces[plain_scenario], policy_traces_by_seed[plot_seed][plain_scenario],
                     plain_scenario, os.path.join(PLOTS_DIR, "speed_vs_time.png"))
    plot_cumulative_fuel(baseline_traces[plain_scenario], policy_traces_by_seed[plot_seed][plain_scenario],
                          plain_scenario, os.path.join(PLOTS_DIR, "cumulative_fuel.png"))
    plot_gap(baseline_traces[leader_scenario], policy_traces_by_seed[plot_seed][leader_scenario],
             leader_scenario, os.path.join(PLOTS_DIR, "gap_vs_time.png"))
    plot_learning_curves(args.seeds, os.path.join(PLOTS_DIR, "learning_curves.png"))

    print(f"\nPlots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
