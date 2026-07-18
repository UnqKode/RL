from eco_driving.scripts.train import train_one_seed
from eco_driving.config import TrainConfig
tc = TrainConfig()
train_one_seed(seed=0, total_timesteps=700_000, eval_freq=tc.eval_freq,
                n_eval_episodes=tc.n_eval_episodes, out_subdir='sac_seed0_round4_700k')
