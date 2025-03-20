import argparse
import json
import os
from typing import Dict, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from controllers import CEMMPCController
from dynamics import MLPDynamicsModel
from utils import load_model_from_checkpoint, seed_env, seed_everything


def plot_trajectory_comparison(
    true_trajectory: np.ndarray, predicted_trajectory: np.ndarray, state_labels: List[str], save_path: str = None
):
    """Plot comparison between true and predicted trajectories."""
    n_steps, n_dims = true_trajectory.shape
    time_steps = np.arange(n_steps)

    # Set up the plot style
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 3 * n_dims))
    if n_dims == 1:
        axes = [axes]

    for dim, (ax, label) in enumerate(zip(axes, state_labels)):
        # Plot true trajectory
        ax.plot(time_steps, true_trajectory[:, dim], label="True", color="blue", linestyle="-")
        # Plot predicted trajectory
        ax.plot(time_steps, predicted_trajectory[:, dim], label="Predicted", color="red", linestyle="--")

        ax.set_xlabel("Time Step")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

    plt.show()


def predict_trajectory(
    dynamics_model: MLPDynamicsModel, initial_state: np.ndarray, actions: np.ndarray, horizon: int
) -> np.ndarray:
    with torch.no_grad():
        # Add batch dimension to initial state if not present
        if initial_state.ndim == 1:
            initial_state = initial_state[None, :]  # Shape: [1, state_dim]
        initial_state = torch.FloatTensor(initial_state)

        # Add batch dimension to actions if not present
        if actions.ndim == 2:  # [horizon, action_dim]
            actions = actions[None, :, :]  # Shape: [1, horizon, action_dim]
        actions = torch.FloatTensor(actions)

        # Get predictions
        predicted_states = dynamics_model.predict_n_steps(initial_state, actions, horizon)

        # Remove batch dimension for output
        predicted_states = predicted_states.squeeze(0)

    return predicted_states.numpy()


def collect_evaluation_rollouts(
    env: gym.Env,
    controller: CEMMPCController,
    dynamics_model: MLPDynamicsModel,
    num_rollouts: int,
    rollout_horizon: int,  # Renamed from horizon to rollout_horizon
    max_attempts_per_rollout: int = 5,
) -> List[Dict]:
    """Collect successful rollouts using the controller.

    Args:
        env: Gymnasium environment
        controller: CEM-MPC controller
        dynamics_model: Trained dynamics model
        num_rollouts: Number of successful rollouts to collect
        rollout_horizon: Length of each rollout during collection
        max_attempts_per_rollout: Maximum attempts to get a successful rollout
    """
    rollouts = []

    for i in range(num_rollouts):
        print(f"Collecting rollout {i+1}/{num_rollouts}")

        for attempt in range(max_attempts_per_rollout):
            initial_state, _ = env.reset(seed=42)
            states = [initial_state.copy()]
            actions = []

            terminated = truncated = False
            for step in range(rollout_horizon):
                action, _ = controller.plan(states[-1])
                next_state, reward, terminated, truncated, _ = env.step(action)

                states.append(next_state.copy())
                actions.append(action.copy())

                if terminated or truncated:
                    break

            if len(states) > rollout_horizon // 2:
                rollout_data = {
                    "initial_state": initial_state,
                    "states": np.array(states),
                    "actions": np.array(actions),
                    "length": len(states) - 1,  # Changed from 'horizon' to 'length'
                }
                rollouts.append(rollout_data)
                print(f"  Success after {attempt+1} attempts")
                break

            if attempt == max_attempts_per_rollout - 1:
                print(f"  Failed to collect rollout after {max_attempts_per_rollout} attempts")

    return rollouts


def save_rollouts(rollouts: List[Dict], save_path: str):
    """Save rollouts to a file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_rollouts = []
    for rollout in rollouts:
        serializable_rollout = {
            "initial_state": rollout["initial_state"].tolist(),
            "states": rollout["states"].tolist(),
            "actions": rollout["actions"].tolist(),
            "horizon": rollout["horizon"],
        }
        serializable_rollouts.append(serializable_rollout)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(serializable_rollouts, f)
    print(f"Saved rollouts to {save_path}")


def load_rollouts(load_path: str) -> List[Dict]:
    """Load rollouts from a file."""
    with open(load_path, "r") as f:
        serialized_rollouts = json.load(f)

    # Convert lists back to numpy arrays
    rollouts = []
    for rollout in serialized_rollouts:
        deserialized_rollout = {
            "initial_state": np.array(rollout["initial_state"]),
            "states": np.array(rollout["states"]),
            "actions": np.array(rollout["actions"]),
            "horizon": rollout["length"],
        }
        rollouts.append(deserialized_rollout)

    return rollouts


def evaluate_dynamics_on_rollouts(
    dynamics_model: MLPDynamicsModel,
    rollouts: List[Dict],
    eval_horizon: int,  # Added evaluation horizon parameter
    plot_dir: str = None,
):
    """Evaluate dynamics model predictions on collected rollouts.

    Args:
        dynamics_model: Trained dynamics model
        rollouts: List of collected rollouts
        eval_horizon: Horizon to use for evaluation (can be different from collection horizon)
        plot_dir: Directory to save plots
    """
    for i, rollout in enumerate(rollouts):
        initial_state = rollout["initial_state"]
        true_states = rollout["states"]
        actions = rollout["actions"]

        # Use minimum of available actions and eval_horizon
        horizon_to_use = min(len(actions), eval_horizon)
        actions_to_use = actions[:horizon_to_use]
        true_states_to_use = true_states[: horizon_to_use + 1]  # +1 for initial state

        # Get model predictions
        predicted_states = predict_trajectory(dynamics_model, initial_state, actions_to_use, horizon_to_use)

        state_dim = true_states.shape[1]
        state_labels = [f"State Dimension {i}" for i in range(state_dim)]

        if plot_dir:
            save_path = os.path.join(plot_dir, f"rollout_{i+1}_horizon_{horizon_to_use}.png")
        else:
            save_path = None

        plot_trajectory_comparison(true_states_to_use, predicted_states, state_labels, save_path)

        # Print error metrics
        mse = np.mean((true_states_to_use - predicted_states) ** 2)
        print(f"Rollout {i+1} MSE over {horizon_to_use} steps: {mse:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Collect and evaluate rollouts")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--env_name", type=str, default="InvertedPendulum-v5", help="Gymnasium environment name")
    parser.add_argument("--rollout_horizon", type=int, default=200, help="Horizon for collecting rollouts")
    parser.add_argument("--eval_horizon", type=int, default=100, help="Horizon for evaluating dynamics predictions")
    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of rollouts to collect")
    parser.add_argument(
        "--rollouts_path", type=str, default="data/evaluation_rollouts.json", help="Path to save/load rollouts"
    )
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument(
        "--collect_new_rollouts", action="store_true", help="Whether to collect new rollouts or use existing ones"
    )
    args = parser.parse_args()

    # Create environment
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Seed everything
    seed_everything(42)
    seed_env(env, 42)

    # Load dynamics model
    dynamics_model = load_model_from_checkpoint(args.checkpoint, state_dim, action_dim)

    if args.collect_new_rollouts:
        # Create controller
        controller = CEMMPCController(
            # dynamics_model=dynamics_model.model,
            env=env,
            cem_horizon=30,  # Controller planning horizon (different from rollout horizon)
        )

        # Collect rollouts
        rollouts = collect_evaluation_rollouts(
            env,
            controller,
            dynamics_model,
            args.num_rollouts,
            args.rollout_horizon,  # Use rollout_horizon for collection
        )

        save_rollouts(rollouts, args.rollouts_path)
    else:
        rollouts = load_rollouts(args.rollouts_path)

    os.makedirs(args.plot_dir, exist_ok=True)

    # Evaluate dynamics model using eval_horizon
    evaluate_dynamics_on_rollouts(
        dynamics_model, rollouts, args.eval_horizon, args.plot_dir  # Use eval_horizon for evaluation
    )

    env.close()


if __name__ == "__main__":
    main()
