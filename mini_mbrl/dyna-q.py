from utils import *
from envs.gridworld import GridWorldEnv
import time
import matplotlib.pyplot as plt


class DynaQAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.height * env.width, env.action_space.n))
        self.episode_rewards = []
        self.episode_lengths = []
    
    def select_action(self, state, epsilon=0.1):
        state_idx = self.state_to_index(state)
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state_idx])



if __name__ == "__main__":
    env = GridWorldEnv()
    agent = DynaQAgent(env)