from policies import RandomPolicy
from rollout_buffer import RolloutBuffer
from utils import seed_everything
from models import MLP
import gymnasium as gym

import os
def main():
    env_name = "InvertedPendulum-v4" # continuous action space version of Pendulum
    env = gym.make(env_name, render_mode= None)

    policy = RandomPolicy(env)
    action = policy()
    buffer = RolloutBuffer()
    state = env.reset(seed=42)
    seed_everything(42, env)

    episodes_nums = 100
    steps = 100
    for episode in range(episodes_nums):
        state = env.reset()
        for step in range(steps):
            action = policy()
            next_state, reward, terminated, truncated,info = env.step(action)
            buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            if terminated:
                break
        print(f"Episode {episode} done")
    save_buffer = True
    if save_buffer:
        save_path = "data"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        buffer.save(os.path.join("data", "buffer.npy"))
        print("Buffer saved")

def load_buffer():
    buffer = RolloutBuffer()
    buffer.load(os.path.join("data", "buffer.npy"))
    print("Buffer loaded")
    print("Buffer size:", len(buffer))
    print("Buffer max size:", buffer.max_size)
    sample = buffer.sample(batch_size=1)
    print(sample)    

if __name__ == "__main__":
    # main()
    load_buffer()
