"""Quick diagnostic sweeps to isolate why SAC got stuck idling: leader-runaway
reward confound vs. VecNormalize reward-normalization scale vs. init action bias.
Trains short budgets (fast) and reports the eval-metrics trend.
"""
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from eco_driving.config import EnvConfig, TrainConfig
from eco_driving.envs import EcoDrivingEnv


def make_env(cfg):
    def _init():
        return Monitor(EcoDrivingEnv(cfg))
    return _init


def quick_eval(model, vecnorm, cfg, n_episodes=6):
    rewards, fuels, times, stops = [], [], [], []
    for ep in range(n_episodes):
        env = EcoDrivingEnv(cfg)
        obs, info = env.reset(seed=1000 + ep)
        done = False
        ep_r = ep_fuel = ep_stop = 0.0
        while not done:
            norm_obs = vecnorm.normalize_obs(obs.reshape(1, -1)).astype(np.float32)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action[0])
            done = term or trunc
            ep_r += r
            ep_fuel += info["fuel_mL"]
            if info["v"] < 0.3:
                ep_stop += 1
        rewards.append(ep_r); fuels.append(ep_fuel); times.append(env.t); stops.append(ep_stop)
    return np.mean(rewards), np.mean(fuels), np.mean(times), np.mean(stops)


def run_trial(name, cfg_overrides, norm_reward, timesteps, seed=0):
    cfg = EnvConfig(**cfg_overrides)
    train_cfg = TrainConfig()

    venv = DummyVecEnv([make_env(cfg)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)

    model = SAC("MlpPolicy", venv, learning_rate=train_cfg.learning_rate,
                buffer_size=50_000, batch_size=train_cfg.batch_size, gamma=train_cfg.gamma,
                tau=train_cfg.tau, train_freq=1, gradient_steps=1,
                learning_starts=train_cfg.learning_starts, ent_coef="auto",
                policy_kwargs=dict(net_arch=list(train_cfg.net_arch)), seed=seed, verbose=0)

    checkpoints = []
    step_chunk = timesteps // 4
    for i in range(4):
        model.learn(total_timesteps=step_chunk, reset_num_timesteps=False, progress_bar=False)
        r, fuel, t, stops = quick_eval(model, venv, cfg)
        checkpoints.append((model.num_timesteps, r, fuel, t, stops))
        print(f"[{name}] t={model.num_timesteps:>7d} eval_R={r:8.2f} fuel={fuel:6.2f} "
              f"time={t:6.1f} stops={stops:6.1f}")
    return checkpoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=24000)
    args = parser.parse_args()

    print("=== Trial A: baseline config (leader p=0.8, norm_reward=True) ===")
    run_trial("A_full", {}, norm_reward=True, timesteps=args.timesteps)

    print("\n=== Trial B: no leader (p_leader=0), norm_reward=True ===")
    run_trial("B_noleader", dict(p_leader=0.0), norm_reward=True, timesteps=args.timesteps)

    print("\n=== Trial C: leader p=0.8, norm_reward=False ===")
    run_trial("C_noRewNorm", {}, norm_reward=False, timesteps=args.timesteps)

    print("\n=== Trial D: no leader, norm_reward=False ===")
    run_trial("D_noleader_noRewNorm", dict(p_leader=0.0), norm_reward=False, timesteps=args.timesteps)
