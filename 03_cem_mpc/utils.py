import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    print(f"Seeding everything with {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)  # Set hash seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # Ensure PyTorch determinism


def seed_env(env, seed: int):
    """
    Seed the environment with a given seed.
    # Still need to set the seed for the environment's reset method
    """
    seed_everything(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Check for Mujoco-specific seeding
    if hasattr(env.unwrapped, "sim"):
        env.unwrapped.sim.model.opt.random_seed = seed
        env.unwrapped.sim.data.qpos = None  # Reset to deterministic state if needed
        env.unwrapped.sim.data.qvel = None

    # Seed any additional RNGs used by wrappers
    if hasattr(env, "seed"):
        env.seed(seed)


def to_torch(x, dtype: torch.dtype, device: torch.device = torch.device("cpu")) -> torch.Tensor:
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


def load_model_from_checkpoint(checkpoint_path: str, state_dim: int, action_dim: int) -> torch.nn.Module:
    """Load a dynamics model from a checkpoint."""
    from dynamics import MLPDynamicsModel

    dynamics_model = MLPDynamicsModel(state_dim, action_dim)
    checkpoint = torch.load(checkpoint_path)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        dynamics_model.model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume checkpoint is the state dict itself
        dynamics_model.model.load_state_dict(checkpoint)

    dynamics_model.model.eval()
    return dynamics_model
