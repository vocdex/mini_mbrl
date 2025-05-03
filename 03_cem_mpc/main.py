"""
Epoch consists of:
1. Dynamics model training: update model_updates_per_epoch times
- Each update uses a random batch of data from the replay buffer
2. Collect data using the controller: steps_per_epoch times
- For each step, the controller plans using the current dynamics model and environment state
- The controller executes the action for up to steps_per_epoch steps and stores the transitions in the replay buffer
- If the environment episode terminates, a new episode is started
"""
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import wandb
from controllers import CEMMPCController
from dynamics import MLPDynamicsModel
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration for the training process."""

    env_name: str
    n_epochs: int = 5
    steps_per_epoch: int = 200
    batch_size: int = 64
    initial_random_steps: int = 1000
    model_updates_per_epoch: int = 10
    replay_buffer_size: int = 10000
    horizon: int = 30  # Planning horizon for CEM-MPC
    num_simulated_trajectories: int = 100  # Number of trajectories for CEM
    num_elites: int = 10  # Number of elite trajectories for CEM
    optim_iterations: int = 5  # Optimization iterations for CEM
    log_to_wandb: bool = True
    project_name: str = "model-based-rl"
    run_name: Optional[str] = None
    checkpoint_dir: str = "checkpoints"


class ReplayBuffer:
    """Generic replay buffer implementation."""

    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        self.buffer.append((state, action, next_state))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]

        states = torch.FloatTensor(np.array([x[0] for x in batch]))
        actions = torch.FloatTensor(np.array([x[1] for x in batch]))
        next_states = torch.FloatTensor(np.array([x[2] for x in batch]))

        return states, actions, next_states

    def __len__(self):
        return len(self.buffer)


class MBRLTrainer:
    """Main trainer class for model-based RL."""

    def __init__(
        self,
        config: TrainingConfig,
        dynamics_model: Any,
        env: Optional[gym.Env] = None,
    ):
        self.config = config
        self.dynamics_model = dynamics_model

        # Create or use provided environment
        self.env = env if env is not None else gym.make(config.env_name)

        # Initialize controller with environment
        self.controller = CEMMPCController(
            dynamics_model=dynamics_model.model,
            env=self.env,
            cem_horizon=config.horizon,
            num_simulated_trajectories=config.num_simulated_trajectories,
            num_elites=config.num_elites,
            optim_iterations=config.optim_iterations,
        )

        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        os.makedirs(config.checkpoint_dir, exist_ok=True)

        if config.log_to_wandb:
            wandb.init(project=config.project_name, name=config.run_name or config.env_name, config=vars(config))

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save a checkpoint of the dynamics model."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch+1}.pth")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.dynamics_model.model.state_dict(),
            "metrics": metrics,
        }

        torch.save(checkpoint, checkpoint_path)

        if self.config.log_to_wandb:
            wandb.save(checkpoint_path)

        print(f"Saved checkpoint for epoch {epoch+1} at {checkpoint_path}")

    def collect_initial_data(self):
        """Collect initial random data for the replay buffer."""
        state, _ = self.env.reset()
        for _ in range(self.config.initial_random_steps):
            action = self.env.action_space.sample()
            next_state, _, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.add(state, action, next_state)

            if terminated or truncated:
                state, _ = self.env.reset()
            else:
                state = next_state

    def train_dynamics_model(self) -> List[float]:
        """Train the dynamics model on a batch of data."""
        losses = []
        for _ in range(self.config.model_updates_per_epoch):
            states, actions, next_states = self.replay_buffer.sample(self.config.batch_size)
            loss = self.dynamics_model.train(states, actions, next_states)
            losses.append(loss)
        return losses

    def run_epoch(self, epoch: int) -> Dict[str, float]:
        """Run a single training epoch."""
        state, _ = self.env.reset()
        epoch_reward = 0
        steps = 0
        episode_rewards = []

        # Train dynamics model
        epoch_losses = self.train_dynamics_model()

        # Collect data using the controller
        episode_reward = 0
        for step in range(self.config.steps_per_epoch):
            # Plan action using the controller
            action, _ = self.controller.plan(state)

            # Execute action in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.add(state, action, next_state)

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state, _ = self.env.reset()
            else:
                state = next_state

        metrics = {
            "epoch": epoch + 1,
            "average_episode_reward": np.mean(episode_rewards) if episode_rewards else episode_reward,
            "steps": steps,
            "model_loss": np.mean(epoch_losses),
            "num_episodes": len(episode_rewards),
        }

        if self.config.log_to_wandb:
            wandb.log(metrics)

        # Save checkpoint after each epoch
        self.save_checkpoint(epoch, metrics)

        return metrics

    def train(self) -> Tuple[Any, Any, Dict[str, List[float]]]:
        """Main training loop."""
        print("Collecting initial random data...")
        self.collect_initial_data()

        print("Starting training...")
        metrics_history = {"rewards": [], "steps": [], "model_losses": [], "num_episodes": []}

        for epoch in tqdm(range(self.config.n_epochs)):
            metrics = self.run_epoch(epoch)

            metrics_history["rewards"].append(metrics["average_episode_reward"])
            metrics_history["steps"].append(metrics["steps"])
            metrics_history["model_losses"].append(metrics["model_loss"])
            metrics_history["num_episodes"].append(metrics["num_episodes"])

            print(
                f"Epoch {epoch+1}: "
                f"Avg Reward = {metrics['average_episode_reward']:.2f}, "
                f"Steps = {metrics['steps']}, "
                f"Number of Episodes = {metrics['num_episodes']}, "
                f"Model Loss = {metrics['model_loss']:.4f}"
            )

        self.env.close()
        if self.config.log_to_wandb:
            wandb.finish()

        return self.dynamics_model, self.controller, metrics_history


def main():
    # Example configuration
    config = TrainingConfig(
        env_name="InvertedPendulum-v5",
        n_epochs=50,
        steps_per_epoch=200,
        batch_size=64,
        horizon=30,
        num_simulated_trajectories=100,
        num_elites=10,
        optim_iterations=5,
        project_name="model-based-rl",
        run_name="CEM-MPC-InvertedPendulum",
        checkpoint_dir="model_checkpoints",
    )

    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dynamics_model = MLPDynamicsModel(state_dim, action_dim)

    trainer = MBRLTrainer(config, dynamics_model, env)
    dynamics_model, controller, metrics = trainer.train()

    return dynamics_model, controller, metrics


if __name__ == "__main__":
    dynamics_model, controller, metrics = main()
