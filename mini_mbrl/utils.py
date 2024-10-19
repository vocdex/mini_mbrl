import torch
import random
import numpy as np
import os


def seed_everything(seed: int):
    print(f"Seeding everything with {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # This is to ensure that the hash function is deterministic


def seed_env(env, seed: int):
    """
    Seed the environment with a given seed.
    """
    seed_everything(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def to_torch(
    x, dtype: torch.dtype, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Convert a numpy array or a list to a torch tensor.
    """
    if isinstance(x, list):
        x = np.array(x, dtype=np.float32)

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    else:
        print(f"Unknown type {type(x)}")
