from rollout_buffer import RolloutBuffer
import gymnasium as gym
import torch
import os
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def seed_env(env, seed: int):
    seed_everything(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    

class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, *args, **kwargs):
        return self.env.action_space.sample()


class OpenLoopPolicy:
    """This should return an action when called with an observation."""

    def __init__(self, action_sequence, env: gym.Env):
        raise NotImplementedError

    def get_action(self, obs, state, **kwargs):
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = self.layer_norm1(x)
        x = F.leaky_relu(self.fc2(x))
        state_delta = self.fc3(x)
        return state_delta

def main(params):
    # Initialize environment
    env = gym.make(params['env_name'], render_mode=None)
    seed_everything(params['seed'])
    
    # Initialize policy
    policy = RandomPolicy(env)

    # Initialize model
    model = MLP(env.observation_space.shape[0], env.action_space.shape[0])

    # Initialize buffer
    buffer = RolloutBuffer()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Loss criterion
    criterion = torch.nn.MSELoss()
    
    # Seed environment
    seed_env(env, params['seed'])

    # Main loop
    """
    1. Collect data with a random policy
    2. Update model with the collected data (train epoch_num times on the collected data)
    3. Repeat
    """

    episode_nums = 10
    outer_loop = 10
    epoch_num = 10
    batch_size = 64
    
    for i in range(outer_loop):
        print("Starting iteration", i)
        if i > 0:
            model_path = os.path.join(params['log_dir'], f"model_{i-1}.pth")
            model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")

        buffer = collect_data(env, episode_nums, policy, buffer)
        buffer_path = os.path.join(params['log_dir'], f"buffer{i}.pth")
        buffer.save(buffer_path)
        print(f"Buffer saved to {buffer_path}")

        print(f"Training model {i}")
        for epoch in range(epoch_num):
            epoch_loss = 0
            for _ in range(len(buffer) // batch_size):
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = buffer.sample(batch_size)
                
                optimizer.zero_grad()
                pred = model(state_batch, action_batch)
                target = next_state_batch  # Usually, target can be the next state. Adjust if necessary.
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch} Loss: {epoch_loss}")
        
        model_save_path = os.path.join(params['log_dir'], f"model_{i}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    
    return model

def collect_data(env, episode_nums, policy, buffer):
    for episode in range(episode_nums):
        state, info = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
    return buffer


if __name__ == "__main__":
    params = {"env_name": "InvertedPendulum-v4", "seed": 0, "lr": 1e-3, "log_dir": "logs"}
    if not os.path.exists(params['log_dir']):
        os.makedirs(params['log_dir'])
    main(params)
