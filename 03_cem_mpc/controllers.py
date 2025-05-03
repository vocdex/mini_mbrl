from typing import Any, Callable, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from mujoco_wrapper import StatefulMujocoWrapper
from utils import load_model_from_checkpoint


class CEMMPCController:
    """CEM MPC controller supporting both learned and ground truth dynamics."""

    def __init__(
        self,
        env: Any,
        dynamics_model: Optional[Any] = None,
        cem_horizon: int = 30,
        num_simulated_trajectories: int = 100,
        num_elites: int = 10,
        optim_iterations: int = 5,
        reward_function: Optional[Callable] = None,
    ):
        # Make sure environment is properly wrapped
        raw_env = env if isinstance(env, gym.Env) else gym.make(env)
        self.env = raw_env  # Assume env is already wrapped with StatefulMujocoWrapper

        self.dynamics_model = dynamics_model
        self.use_ground_truth = dynamics_model is None

        self.cem_horizon = cem_horizon
        self.num_simulated_trajectories = num_simulated_trajectories
        self.num_elites = num_elites
        self.optim_iterations = optim_iterations

        # Set reward function
        self.compute_reward = self._compute_ground_truth_reward if self.use_ground_truth else self._compute_reward
        self.initial_sigma = 0.1
        self.sigma_decay = 0.9

    def _predict_next_state(self, state, action):
        if self.use_ground_truth:
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            # Save both state and step counter
            original_step_count = self.env.env._elapsed_steps
            original_state = self.env.save_state()

            # Execute step
            next_obs, _, _, _, _ = self.env.step(action)

            # Restore both state and step counter
            self.env.restore_state(original_state)
            self.env.env._elapsed_steps = original_step_count

            return next_obs
        else:
            # Use learned dynamics model
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if isinstance(action, np.ndarray):
                action = torch.FloatTensor(action)

            with torch.no_grad():
                next_state = (
                    self.dynamics_model.predict_1_step(state, action).squeeze(0).detach().cpu().numpy()
                )  # Shape: [state_dim]
                return next_state

    def _compute_ground_truth_reward(self, state, action, next_state):
        original_step_count = self.env.env._elapsed_steps
        original_state = self.env.save_state()

        _, reward, terminated, truncated, _ = self.env.step(action)

        self.env.restore_state(original_state)
        self.env.env._elapsed_steps = original_step_count

        return reward

    def _compute_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        if self.use_ground_truth:
            original_state = self.env.save_state()
            self.env.restore_state(original_state)
            _, reward, _, _, _ = self.env.step(action)
            self.env.restore_state(original_state)
            return reward
        else:
            # Custom reward computation for learned dynamics
            # For InvertedPendulum, we can compute the reward directly
            x, x_dot, theta, theta_dot = next_state
            reward = 1.0  # Base reward for not failing

            # Penalize if the pendulum is far from vertical
            if abs(theta) > 0.2:
                reward = 0.0

            return reward

    def plan(self, current_state, action_dim=None, action_bounds=None):
        if action_dim is None:
            action_dim = self.env.action_space.shape[0]
        if action_bounds is None:
            action_bounds = (self.env.action_space.low[0], self.env.action_space.high[0])

        # Initialize with smaller variance
        mu = np.zeros((self.cem_horizon, action_dim))
        sigma = np.ones((self.cem_horizon, action_dim)) * self.initial_sigma

        best_return = float("-inf")
        best_action = np.zeros(action_dim)
        best_trajectory = None

        for iter in range(self.optim_iterations):
            # Sample and evaluate trajectories
            trajectories = self.sample_trajectories(mu, sigma, action_dim)
            trajectories = np.clip(trajectories, action_bounds[0], action_bounds[1])
            returns = self.evaluate_trajectories(current_state, trajectories)

            # Update statistics
            elite_idx = np.argsort(returns)[-self.num_elites :]
            elite_trajectories = trajectories[elite_idx]

            # Update mean and variance with momentum
            new_mu = np.mean(elite_trajectories, axis=0)
            new_sigma = np.std(elite_trajectories, axis=0) + 1e-5

            # Smooth updates
            mu = 0.9 * mu + 0.1 * new_mu
            sigma = 0.9 * sigma + 0.1 * new_sigma

            # Apply sigma decay
            sigma *= self.sigma_decay

            # Update best trajectory
            if np.max(returns) > best_return:
                best_return = np.max(returns)
                best_idx = np.argmax(returns)
                best_action = trajectories[best_idx][0]
                best_trajectory = trajectories[best_idx]

        return best_action, best_trajectory

    def sample_trajectories(self, mu: np.ndarray, sigma: np.ndarray, action_dim: int) -> np.ndarray:
        """Sample trajectories from a Gaussian distribution."""
        return np.random.normal(mu, sigma, size=(self.num_simulated_trajectories, self.cem_horizon, action_dim))

    def evaluate_trajectories(self, initial_state: np.ndarray, trajectories: np.ndarray) -> np.ndarray:
        returns = np.zeros(len(trajectories))

        for i, actions in enumerate(trajectories):
            total_reward = 0
            state = initial_state
            terminated = False

            for action in actions:
                if terminated:
                    break

                next_state = self._predict_next_state(state, action)
                reward = self.compute_reward(state, action, next_state)
                # Check if next_state violates constraints
                if np.abs(next_state[2]) > 0.2:  # Pendulum angle threshold
                    reward = -100
                    terminated = True

                total_reward += reward
                state = next_state

            returns[i] = total_reward

        return returns


# def run_episode(env, controller, max_steps=1000, render=True):
#     """Run a single episode with the CEM-MPC controller."""
def run_episode(env, controller, max_steps=1000, seed=42):
    """Run a single episode with the CEM-MPC controller."""
    observation, _ = env.reset(seed=seed)
    total_reward = 0
    actual_step = 0  # Track actual steps separately

    while actual_step < max_steps:
        print(f"Step {actual_step}")
        # print(f"Step {actual_step}")
        action, planned_trajectory = controller.plan(observation)

        # Execute action in environment
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        actual_step += 1

        if terminated:
            print("Episode terminated.")
            break
        if truncated:
            print("Episode truncated.")

    print(f"Total Reward: {total_reward}")
    return total_reward


def benchmark_controller(env, controller, num_episodes=10):
    """Benchmark the CEM-MPC controller over multiple episodes."""
    total_rewards = []

    for i in range(num_episodes):
        print(f"Running episode {i}...")
        total_reward = run_episode(env, controller)
        total_rewards.append(total_reward)

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


def main():
    # Create environment
    from gymnasium.wrappers import RecordVideo

    trigger = lambda t: True
    from utils import seed_env, seed_everything

    seed_everything(42)

    env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")
    seed_env(env, 42)
    env = RecordVideo(env, video_folder="./save_videos", episode_trigger=trigger, disable_logger=True)

    env = StatefulMujocoWrapper(env)

    checkpoint_path = "/Users/shuk/Desktop/mini_mbrl/mini_mbrl/03_cem_mpc/model_checkpoints/epoch_1.pth"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dynamics_model = load_model_from_checkpoint(checkpoint_path, state_dim, action_dim)

    # Create controller - parameters tuned for inverted pendulum
    controller = CEMMPCController(
        dynamics_model=dynamics_model,
        env=env,
        cem_horizon=10,
        num_simulated_trajectories=128,  # Increased population size
        num_elites=20,  # Increased elite set
        optim_iterations=3,  # More iterations
    )

    # print("Starting episode...")
    # total_reward = run_episode(env, controller, max_steps=1000, seed=42)
    # print(f"\nEpisode finished with total reward: {total_reward:.2f}")

    # Benchmark controller
    avg_reward = benchmark_controller(env, controller, num_episodes=10)
    print(f"\nAverage reward over 50 episodes: {avg_reward :.2f}")
    env.close()


if __name__ == "__main__":
    main()
