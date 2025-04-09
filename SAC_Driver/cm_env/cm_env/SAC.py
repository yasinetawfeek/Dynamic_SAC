from stable_baselines3 import SAC
# from sbx import SAC
from gym_env_new import CmEnv
import torch.nn as nn
from config import *

path = "/media/yasinetawfeek/069F-E29D/SAC_ipg"
logs = path + "/logs"

policy_kwargs = dict(
    # net_arch=[512, 256, 128],
    net_arch = [256, 256],
    activation_fn=nn.LeakyReLU
)

env = CmEnv(build_flag=False)

model = SAC(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    batch_size=512,
    learning_rate=3e-4,
    device="cuda",
    verbose=1,
    tensorboard_log=logs
)

# model = SAC.load("/media/yasinetawfeek/069F-E29D/SAC_ipg/models/SAC_ipg_models_4.zip", env=env)

counter = 1

while True:
    model.learn(total_timesteps=TIMESTEPS_PER_MODEL_SAVE, reset_num_timesteps=False, progress_bar=True, tb_log_name="SAC_FUNC_Final_A1=10_A2=0_A3=400000_DISC=0.999_CHECKDISS=0.0025")
    model.save( path + f"/models/Final_Function/A1=10_A2=0_A3=400000_DISC=0.999_CHECKDISS=0.0025/{counter * TIMESTEPS_PER_MODEL_SAVE}")
    counter += 1
