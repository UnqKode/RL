"""Train SAC on the eco-driving env: Monitor -> DummyVecEnv -> VecNormalize.

Usage:
    python -m eco_driving.scripts.train --seed 0 --timesteps 400000
    python -m eco_driving.scripts.train --all-seeds        # trains seeds 0,1,2 sequentially
"""
import argparse
import csv
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..config import EnvConfig, TrainConfig
from ..envs import EcoDrivingEnv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")


def make_env(cfg: EnvConfig, monitor_path: str = None):
    def _init():
        env = EcoDrivingEnv(cfg)
        env = Monitor(env, filename=monitor_path)
        return env
    return _init


class MetricsLogger(BaseCallback):
    """Deterministic rollout on eval_env every eval_freq steps; logs reward, fuel,
    travel time, and stop count (v < 0.3 m/s) to a CSV for later inspection/plots.
    """

    def __init__(self, eval_env: VecNormalize, n_episodes: int, eval_freq: int, csv_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_episodes = n_episodes
        self.eval_freq = eval_freq
        self.csv_path = csv_path

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        rewards, fuels, times, stops, maxjerks = [], [], [], [], []
        for _ in range(self.n_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_r, ep_fuel, ep_stop, ep_maxjerk, ep_t = 0.0, 0.0, 0, 0.0, 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done_arr, infos = self.eval_env.step(action)
                done = bool(done_arr[0])
                info = infos[0]
                ep_r += float(reward[0])
                ep_fuel += info.get("fuel_mL", 0.0)
                ep_t = info.get("t", ep_t)
                if info.get("v", 1.0) < 0.3:
                    ep_stop += 1
                ep_maxjerk = max(ep_maxjerk, abs(info.get("jerk", 0.0)))
            rewards.append(ep_r); fuels.append(ep_fuel); times.append(ep_t)
            stops.append(ep_stop); maxjerks.append(ep_maxjerk)

        row = dict(timesteps=self.num_timesteps,
                   mean_reward=np.mean(rewards),
                   mean_fuel_mL=np.mean(fuels),
                   mean_travel_time_s=np.mean(times),
                   mean_stop_steps=np.mean(stops),
                   mean_max_abs_jerk=np.mean(maxjerks))
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        if self.verbose:
            print(f"[metrics] t={self.num_timesteps} R={row['mean_reward']:.1f} "
                  f"fuel={row['mean_fuel_mL']:.1f}mL time={row['mean_travel_time_s']:.1f}s "
                  f"stops={row['mean_stop_steps']:.1f} maxjerk={row['mean_max_abs_jerk']:.2f}")
        return True


def train_one_seed(seed: int, total_timesteps: int, eval_freq: int, n_eval_episodes: int,
                    out_subdir: str = None, verbose: int = 1):
    env_cfg = EnvConfig(seed=seed)
    train_cfg = TrainConfig()

    subdir = out_subdir or f"sac_seed{seed}"
    model_dir = os.path.join(MODELS_DIR, subdir)
    os.makedirs(model_dir, exist_ok=True)

    train_monitor_path = os.path.join(model_dir, "train_monitor.csv")
    eval_monitor_path = os.path.join(model_dir, "eval_monitor.csv")
    metrics_csv_path = os.path.join(model_dir, "train_metrics.csv")

    train_venv = DummyVecEnv([make_env(env_cfg, train_monitor_path)])
    train_venv = VecNormalize(train_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_venv = DummyVecEnv([make_env(env_cfg, eval_monitor_path)])
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    model = SAC(
        "MlpPolicy",
        train_venv,
        learning_rate=train_cfg.learning_rate,
        buffer_size=train_cfg.buffer_size,
        batch_size=train_cfg.batch_size,
        gamma=train_cfg.gamma,
        tau=train_cfg.tau,
        train_freq=train_cfg.train_freq,
        gradient_steps=train_cfg.gradient_steps,
        learning_starts=train_cfg.learning_starts,
        ent_coef=train_cfg.ent_coef,
        policy_kwargs=dict(net_arch=list(train_cfg.net_arch)),
        seed=seed,
        verbose=verbose,
    )

    eval_callback = EvalCallback(
        eval_venv,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    metrics_logger = MetricsLogger(eval_venv, n_episodes=n_eval_episodes, eval_freq=eval_freq,
                                    csv_path=metrics_csv_path, verbose=verbose)

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, metrics_logger],
                progress_bar=False)

    model.save(os.path.join(model_dir, "model_final.zip"))
    train_venv.save(os.path.join(model_dir, "vecnormalize.pkl"))

    print(f"[seed {seed}] training complete. Artifacts in {model_dir}")
    return model_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=None)
    args = parser.parse_args()

    train_cfg = TrainConfig()
    timesteps = args.timesteps or train_cfg.total_timesteps
    eval_freq = args.eval_freq or train_cfg.eval_freq
    n_eval_episodes = args.n_eval_episodes or train_cfg.n_eval_episodes

    seeds = list(train_cfg.seeds) if args.all_seeds else [args.seed]
    for seed in seeds:
        train_one_seed(seed, timesteps, eval_freq, n_eval_episodes)


if __name__ == "__main__":
    main()
