from stable_baselines3 import SAC
# from sbx import SAC
from gym_env_new import CmEnv
import torch.nn as nn
from config import *
import pandas as pd

path = "/media/yasinetawfeek/069F-E29D/SAC_ipg"
logs = path + "/logs"

policy_kwargs = dict(
    # net_arch=[512, 256, 128],
    net_arch = [256, 256],
    activation_fn=nn.ReLU,
    # log_std_init=-0.5
)

env = CmEnv(build_flag=False)

model = SAC(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    batch_size=256,
    learning_rate=3e-4,
    buffer_size=10000000,
    learning_starts=1000,
    gamma=0.99,
    use_sde=True,
    ent_coef="auto",
    device="cuda",
    verbose=1,
    tensorboard_log=logs
)

model = SAC.load("/media/yasinetawfeek/069F-E29D/SAC_ipg/models/Final_Function/A1=1_A2=0_A3=0_DISC=0.999_CHECKDISS=0.001_NoRandom_Spawn_SilverstonFinal/600000.zip", env=env)
# model = SAC.load("/media/yasinetawfeek/069F-E29D/SAC_ipg/models/Final_Function/A1=1_A2=0_A3=0_DISC=1_CHECKDISS=0.001_Random_SilverstoneFinalChild/1100000", env=env)

counter = 1
while True:
    model.learn(total_timesteps=TIMESTEPS_PER_MODEL_SAVE, reset_num_timesteps=False, tb_log_name="DissertationDriver", progress_bar=True)
    model.save( path + f"/models/Final_Function/DissertationDriver/{counter * TIMESTEPS_PER_MODEL_SAVE}")
    df = pd.DataFrame(env.checkpoint_history_bank)
    df.to_csv(path + f"/models/Final_Function/DissertationDriver/checkpoint_bank")
    counter += 1
