import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


class BaseAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.3):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # Initialize Q-table
        self.Q = np.zeros((env.height, env.width, env.action_space.n), dtype=np.float32)
        self.rewards_per_episode = []
        self.steps_per_episode = []


    def action(self, state, epsilon=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample() 
        else:
            # Choose the action that maximizes the Q-value
            return np.argmax(self.Q[state[0], state[1]]) 
    
    def learn(self, n_episodes):
        raise NotImplementedError("You need to implement the learn method")
    


    def visualize_q_values(self, agent_type):
        actions = ['Up', 'Right', 'Down', 'Left']
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()
        
        for i in range(self.env.action_space.n):
            ax = axs[i]
            im = ax.imshow(self.Q[:, :, i], cmap='coolwarm', aspect='auto')
            ax.set_title(f"Action: {actions[i]}")
            for y in range(self.Q.shape[0]):
                for x in range(self.Q.shape[1]):
                    ax.text(x, y, f'{self.Q[y, x, i]:.2f}', ha='center', va='center', color='black')
            fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        if not os.path.exists("dyna_plots"):
            os.makedirs("dyna_plots")
        plt.savefig(f"dyna_plots/{agent_type}_q_values.png")
        plt.show()
        return fig
    
    def visualize_policy(self, agent_type):
        fig, ax = plt.subplots(figsize=(6, 6))
        policy = np.argmax(self.Q, axis=2)
        grid_height, grid_width = policy.shape
        
        # Create a grid of arrows
        directions = {0: (0, -0.4), 1: (0.4, 0), 2: (0, 0.4), 3: (-0.4, 0)}  # Up, Right, Down, Left
        direction_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        
        ax.imshow(policy, cmap='gray', vmin=0, vmax=3)
        ax.set_title("Policy")
        
        for y in range(grid_height):
            for x in range(grid_width):
                action = policy[y, x]
                dx, dy = directions[action]
                ax.text(x, y, direction_arrows[action], fontsize=16, ha='center', va='center', color='red')
        
        plt.xticks(np.arange(grid_width))
        plt.yticks(np.arange(grid_height))
        plt.grid(True)

        if not os.path.exists("dyna_plots"):
            os.makedirs("dyna_plots")
        plt.savefig(f"dyna_plots/{agent_type}_policy.png")
        plt.show()
        return fig


class SARSAAgent(BaseAgent):
    """Update rule:
    Q(s_t, a_t) <- Q(s_t, a_t)+ alpha * (R_t+1 + gamma* Q(s_t+1, a_t+1) - Q(s_t, a_t)
    (On-policy TD control)
    We need state, action, reward, next_state, next_action to update the Q-value. Just do an action and update the Q-value.
    """
    def learn(self, n_episodes):
        episode_rewards = []
        episode_steps = []
        
        for _ in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            action = self.action(state)
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action = self.action(next_state)
                target = reward + self.discount_factor * self.Q[next_state[0], next_state[1], next_action]
                self.Q[state[0], state[1], action] += self.learning_rate * (target - self.Q[state[0], state[1], action])
                state, action = next_state, next_action
                
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
        
        return episode_rewards, episode_steps



class QLearningAgent(BaseAgent):
    """Update rule:
    Q(s_t, a_t) <- Q(s_t, a_t)+ alpha * (R_t+1 + gamma* max_a Q(s_t+1, a) - Q(s_t, a_t)
    (Off-policy TD control)
    Apply a greedy policy to update the Q-value"""
    def learn(self, n_episodes):
        episode_rewards = []
        episode_steps = []
        
        for _ in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action = self.action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                target = reward + self.discount_factor * np.max(self.Q[next_state[0], next_state[1]])
                self.Q[state[0], state[1], action] += self.learning_rate * (target - self.Q[state[0], state[1], action])
                state = next_state
                
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
        
        return episode_rewards, episode_steps


class DynaQAgent(BaseAgent):
    """Assume the environment is deterministic. State and action space is discrete. Samples from real environment
    is used to update the Q-value. The model is updated with the real environment samples. The model is used to generate
    simulated samples to update the Q-value. The model is updated with the simulated samples for n_planning_steps.
    So, Q-value is updated with real samples and simulated samples."""

    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, n_planning_steps=5):
        super().__init__(env, learning_rate, discount_factor, epsilon)
        self.model = {}
        self.n_planning_steps = n_planning_steps


    def learn(self, n_episodes):
        episode_rewards = []
        episode_steps = []

        for _ in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action = self.action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                target = reward + self.discount_factor * np.max(self.Q[next_state[0], next_state[1]])
                self.Q[state[0], state[1], action] += self.learning_rate * (target - self.Q[state[0], state[1], action])
                self.model_update(state, action, next_state, reward)
                self.planning(self.n_planning_steps)
                state = next_state
                # self.env.render()
                # time.sleep(0.5)
                
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)

        return episode_rewards, episode_steps

    def model_update(self, state, action, next_state, reward):
        self.model[(state[0], state[1], action)] = (next_state, reward)

    def planning(self, n_planning_steps):
        for _ in range(n_planning_steps):
            state = (np.random.randint(0, self.env.observation_space.high[0]), 
                     np.random.randint(0, self.env.observation_space.high[1]))
            action = np.random.randint(0, self.env.action_space.n)
            
            if (state[0], state[1], action) in self.model:
                next_state, reward = self.model[(state[0], state[1], action)]
                target = reward + self.discount_factor * np.max(self.Q[next_state[0], next_state[1]])
                self.Q[state[0], state[1], action] += self.learning_rate * (target - self.Q[state[0], state[1], action])


def plot_metrics(episode_rewards, episode_steps, agent_type):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plot rewards
    ax[0].plot(episode_rewards)
    ax[0].set_title(f"{agent_type} - Episode Rewards")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Total Reward")

    # Plot steps
    ax[1].plot(episode_steps)
    ax[1].set_title(f"{agent_type} - Episode Steps")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Number of Steps")

    plt.tight_layout()
    if not os.path.exists("dyna_plots"):
        os.makedirs("dyna_plots")
    plt.show()
    fig.savefig(f"dyna_plots/{agent_type}_metrics.png") 



def compare_planning_steps(n_planning_steps=[0, 5, 50]):
    n_episodes = 200
    from envs.gridworld import GridWorldEnv

    env = GridWorldEnv(height=10, width=10, start_position=(0, 0), goal_positions=[(9, 9), (9, 8)], trap_positions=[(5, 4), (5, 9)])

    all_episode_steps = {}

    for n in n_planning_steps:
        agent = DynaQAgent(env, n_planning_steps=n)
        _, steps_per_episode = agent.learn(n_episodes)
        all_episode_steps[n] = steps_per_episode

    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 6))
    for n in n_planning_steps:
        ax.plot(all_episode_steps[n], label=f"{n} planning steps")
    ax.set_title("Dyna-Q - Episode Steps")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number of Steps")

    ax.legend()

    # Save plot in "dyna_plots" directory
    if not os.path.exists("dyna_plots"):
        os.makedirs("dyna_plots")
    plt.savefig("dyna_plots/dyna_q_episode_steps_comparison.png")
    plt.show()
  
# Test the agents
if __name__ =="__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="gridworld")
    argparser.add_argument("--height", type=int, default=10)
    argparser.add_argument("--width", type=int, default=10)
    argparser.add_argument("--start_position", type=str, default="(0,0)")
    argparser.add_argument("--goal_positions", type=str, default="(9,9),(9,8)")
    argparser.add_argument("--trap_positions", type=str, default="(5,4),(5,9)")
    argparser.add_argument("--agent", type=str, default="dyna_q", help="sarsa, q_learning, dyna_q")
    argparser.add_argument("--n_episodes", type=int, default=200)
    argparser.add_argument("--n_planning_steps", type=int, default=5)
    args = argparser.parse_args()

    from ..envs.gridworld import GridWorldEnv
    env = GridWorldEnv(height=args.height, width=args.width, start_position=eval(args.start_position), trap_positions=eval(args.trap_positions), goal_positions=eval(args.goal_positions))
    if args.agent == "sarsa":
        agent = SARSAAgent(env)
    elif args.agent == "q_learning":
        agent = QLearningAgent(env)
    elif args.agent == "dyna_q":
        agent = DynaQAgent(env, n_planning_steps=args.n_planning_steps)
    else:
        raise ValueError("Invalid agent type")
    
    episode_rewards, episode_steps = agent.learn(args.n_episodes)
    if args.n_planning_steps != 0:
        plot_metrics(episode_rewards, episode_steps, agent_type=args.agent+"_"+str(args.n_planning_steps))
        agent.visualize_q_values(agent_type=args.agent+"_"+str(args.n_planning_steps))
        agent.visualize_policy(agent_type=args.agent+"_"+str(args.n_planning_steps))
    else:
        plot_metrics(episode_rewards, episode_steps, agent_type=args.agent)
        agent.visualize_q_values(agent_type=args.agent)
        agent.visualize_policy(agent_type=args.agent)