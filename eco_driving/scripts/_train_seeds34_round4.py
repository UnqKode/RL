from eco_driving.scripts.train import train_one_seed
from eco_driving.config import TrainConfig
tc = TrainConfig()
for seed in [3, 4]:
    train_one_seed(seed=seed, total_timesteps=tc.total_timesteps, eval_freq=tc.eval_freq,
                    n_eval_episodes=tc.n_eval_episodes, out_subdir=f'sac_seed{seed}_round4')
