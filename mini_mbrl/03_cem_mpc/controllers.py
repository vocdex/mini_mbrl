from typing import Any, Callable, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch


class CEMMPCController:
    """CEM MPC controller for a learned dynamics model.

    Args:
        dynamics_model: The learned dynamics model
        env: Gymnasium environment instance or environment ID string
        cem_horizon: Planning horizon for CEM
        num_simulated_trajectories: Number of trajectories to sample
        num_elites: Number of elite trajectories to select
        optim_iterations: Number of optimization iterations for CEM
        reward_function: Optional custom reward function
    """

    def __init__(
        self,
        dynamics_model: Any,
        env: Any,
        cem_horizon: int = 30,
        num_simulated_trajectories: int = 100,
        num_elites: int = 10,
        optim_iterations: int = 5,
        reward_function: Optional[Callable] = None,
    ):
        self.dynamics_model = dynamics_model
        self.env = env if isinstance(env, gym.Env) else gym.make(env)
        self.cem_horizon = cem_horizon
        self.num_simulated_trajectories = num_simulated_trajectories
        self.num_elites = num_elites
        self.optim_iterations = optim_iterations

        # Store initial environment state for reward computation
        self.initial_env_state = self._get_env_state()

        # Set reward function
        if reward_function is not None:
            self.compute_reward = reward_function
        else:
            # Try to determine which reward computation method to use
            if hasattr(self.env.unwrapped, "set_state") and hasattr(self.env.unwrapped, "state"):
                self.compute_reward = self._compute_reward_with_state_setting
            else:
                # Default to a simple reward based on state distance to goal
                self.compute_reward = self._compute_default_reward

    def _get_env_state(self) -> Any:
        """Get the current environment state in a way that can be restored later."""
        if hasattr(self.env.unwrapped, "state"):
            return self.env.unwrapped.state.copy()
        elif hasattr(self.env.unwrapped, "_save_state"):
            return self.env.unwrapped._save_state()
        return None

    def _set_env_state(self, state: Any):
        """Set the environment state."""
        if hasattr(self.env.unwrapped, "set_state"):
            self.env.unwrapped.set_state(state)
        elif hasattr(self.env.unwrapped, "state"):
            self.env.unwrapped.state = state.copy()
        elif hasattr(self.env.unwrapped, "_restore_state"):
            self.env.unwrapped._restore_state(state)

    def _compute_reward_with_state_setting(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray
    ) -> float:
        """Compute reward using environment's native reward function with state setting."""
        # Save current state
        old_state = self._get_env_state()

        # Set environment to the current state
        self._set_env_state(state)

        # Get reward from environment
        _, reward, _, _, _ = self.env.step(action)

        # Restore original state
        if old_state is not None:
            self._set_env_state(old_state)

        return reward

    def _compute_default_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """Default reward function based on state distance to goal state.
        This is a simple example and should be customized based on the environment.
        """
        # For basic control tasks, penalize distance from zero state
        state_cost = np.sum(next_state**2)
        action_cost = 0.1 * np.sum(action**2)  # Small action penalty
        return -(state_cost + action_cost)

    def plan(
        self,
        current_state: np.ndarray,
        action_dim: Optional[int] = None,
        action_bounds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plan actions using CEM."""
        # If action dimensions not provided, get from environment
        if action_dim is None:
            action_dim = self.env.action_space.shape[0]
        if action_bounds is None:
            action_bounds = (self.env.action_space.low[0], self.env.action_space.high[0])

        current_state = torch.FloatTensor(current_state)
        mu = np.zeros((self.cem_horizon, action_dim))
        sigma = np.ones((self.cem_horizon, action_dim))

        best_return = float("-inf")
        best_action = np.zeros(action_dim)
        best_trajectory = np.zeros((self.cem_horizon, action_dim))

        for _ in range(self.optim_iterations):
            trajectories = self.sample_trajectories(mu, sigma, action_dim)
            assert trajectories.shape == (self.num_simulated_trajectories, self.cem_horizon, action_dim)

            trajectories = np.clip(trajectories, action_bounds[0], action_bounds[1])
            returns = self.evaluate_trajectories(current_state, trajectories)

            if np.max(returns) > best_return:
                best_return = np.max(returns)
                best_action = trajectories[np.argmax(returns)][0]
                best_trajectory = trajectories[np.argmax(returns)]

            elite_idx = np.argsort(returns)[-self.num_elites :]
            elite_sequences = trajectories[elite_idx]

            mu = np.mean(elite_sequences, axis=0)
            sigma = np.std(elite_sequences, axis=0) + 1e-5

        return best_action, best_trajectory

    def sample_trajectories(self, mu: np.ndarray, sigma: np.ndarray, action_dim: int) -> np.ndarray:
        """Sample trajectories from a Gaussian distribution."""
        return np.random.normal(mu, sigma, size=(self.num_simulated_trajectories, self.cem_horizon, action_dim))

    def evaluate_trajectories(self, initial_state: torch.Tensor, trajectories: np.ndarray) -> np.ndarray:
        """Evaluate trajectories using the dynamics model and reward function."""
        returns = np.zeros(len(trajectories))

        for i, actions in enumerate(trajectories):
            total_reward = 0
            state = initial_state

            for action in actions:
                action = torch.FloatTensor(action)
                with torch.no_grad():
                    next_state = self.dynamics_model(state, action)

                # Convert tensors to numpy for reward computation
                state_np = state.cpu().numpy()
                action_np = action.cpu().numpy()
                next_state_np = next_state.cpu().numpy()

                reward = self.compute_reward(state_np, action_np, next_state_np)
                total_reward += reward
                state = next_state

            returns[i] = total_reward

        return returns

    def set_custom_reward_function(self, reward_fn: Callable):
        """Set a custom reward function.

        Args:
            reward_fn: Callable that takes (state, action, next_state) and returns a reward
        """
        self.compute_reward = reward_fn
