import random
from collections import deque

import numpy as np
import torch


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

    def sample(self, batch_size: int = 64):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        mini_batch = random.sample(self.buffer, k=batch_size)

        state_batch = to_torch([elem["state"] for elem in mini_batch], dtype=torch.float32, device=self.device)
        action_batch = to_torch([elem["action"] for elem in mini_batch], dtype=torch.float32, device=self.device)
        reward_batch = to_torch([elem["reward"] for elem in mini_batch], dtype=torch.float32, device=self.device)
        next_state_batch = to_torch(
            [elem["next_state"] for elem in mini_batch], dtype=torch.float32, device=self.device
        )
        done_batch = to_torch([elem["done"] for elem in mini_batch], dtype=torch.float32, device=self.device)

        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        info = {
            "size": self.size,
            "max_size": self.max_size,
            "buffer": self.buffer,
        }
        np.save(path, info)

    def load(self, path: str):
        info = np.load(path, allow_pickle=True)
        print(info)
        self.size = info["size"]
        self.max_size = info["max_size"]
        self.buffer = info["buffer"]
