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
    model = SAC.load(os.path.join(model_dir, "best_model.zip"), device="cpu")
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
        if info["v"] < 0.3:
            stop_steps += 1
        max_abs_jerk = max(max_abs_jerk, abs(info["jerk"]))
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
        return driver.act(env.v, gap, rel_v, env.leader.present, dist_to_signal,
                           sig.phase, sig.time_to_change)
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

    print("\n  -- across seeds --")
    for key in ["travel_time_s", "total_fuel_mL", "stop_steps", "max_abs_jerk"]:
        vals = per_seed_means[key]
        print(f"     {key:16s}: {np.mean(vals):8.2f} +/- {np.std(vals):6.2f}  (seed means: {['%.2f' % v for v in vals]})")


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
