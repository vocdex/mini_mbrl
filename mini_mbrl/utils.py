import gymnasium as gym
import numpy as np
import random
import torch


def seed_everything(seed: int, env: gym.Env):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    env.action_space.seed(seed)

