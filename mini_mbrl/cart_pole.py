import gymnasium as gym
import numpy as np


def main():
    env = gym.make("CartPole-v1")
    env.reset(seed=42)  # for initial state
    env.action_space.seed(42)  # for action sampling
    # no more randomness from here on
    for _ in range(100):
        env.render()
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        print("Observation: ", obs)
        print("Reward: ", reward)
        print("Done: ", done)
        print("Truncated: ", truncated)
        print("Info: ", info)
        if done:
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
