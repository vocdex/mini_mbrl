import random
from collections import deque
import numpy as np
import torch
import pickle  # Added for serialization


def to_torch(x, dtype: torch.dtype, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if isinstance(x, list):
        # need to convert to np.array first (it is faster to convert from np.array to torch.Tensor than from list)
        x = np.array(x, dtype=np.float32)

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    else:
        print(f"Unknown type {type(x)}")


class RolloutBuffer:
    def __init__(self, max_size: int = 100000, device: torch.device = torch.device("cpu")):
        super().__init__()
        # deque is a list-like container with fast appends and pops on either end
        self.buffer = deque(maxlen=max_size)
        self.size = 0
        self.max_size = max_size
        self.device = device

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        elem_dict = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        self.buffer.append(elem_dict)
        self.size = np.max([self.size + 1, self.max_size])

    def sample(self, batch_size: int = 64, dynamics_only: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        mini_batch = random.sample(self.buffer, k=batch_size)
        print(type(mini_batch[0]['state']))
        print(mini_batch[0]['state'])

        if dynamics_only:
            return self._prepare_dynamics_batch(mini_batch)
        else:
            state_batch = [elem["state"] for elem in mini_batch]
            action_batch = [elem["action"] for elem in mini_batch]
            reward_batch = [elem["reward"] for elem in mini_batch]
            next_state_batch = [elem["next_state"] for elem in mini_batch]
            done_batch = [elem["done"] for elem in mini_batch]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def _prepare_dynamics_batch(self, mini_batch):
        state_batch = [elem["state"] for elem in mini_batch]
        action_batch = [elem["action"] for elem in mini_batch]
        next_state_batch = [elem["next_state"] for elem in mini_batch]

        return state_batch, action_batch, next_state_batch
    
    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        info = {
            "size": self.size,
            "max_size": self.max_size,
            "buffer": list(self.buffer),  # Convert deque to list for serialization
        }
        with open(path, 'wb') as f:
            pickle.dump(info, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            info = pickle.load(f)
        self.size = info["size"]
        self.max_size = info["max_size"]
        self.buffer = deque(info["buffer"], maxlen=self.max_size)  # Convert list back to deque
