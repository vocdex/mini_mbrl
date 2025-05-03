import json
import os
import time
from datetime import datetime
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch
from controllers import CEMMPCController
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from utils import load_model_from_checkpoint


class BenchmarkFramework:
    """Framework for comparing different RL algorithms."""

    def __init__(self, env_id: str, num_timesteps: int = 100000, num_eval_episodes: int = 10):
        self.env_id = env_id
        self.num_timesteps = num_timesteps
        self.num_eval_episodes = num_eval_episodes
        self.results: Dict[str, Dict] = {}

        # Create results directory
        self.results_dir = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

    def train_model_free(self, algo_class, algo_name: str, **kwargs):
        """Train a model-free algorithm from SB3."""
        print(f"\nTraining {algo_name}...")

        # Create environments
        env = gym.make(self.env_id)
        eval_env = Monitor(gym.make(self.env_id))

        # Initialize algorithm
        model = algo_class("MlpPolicy", env, verbose=1, **kwargs)

        # Track metrics
        start_time = time.time()

        # Train the agent
        model.learn(total_timesteps=self.num_timesteps)

        training_time = time.time() - start_time

        # Evaluate
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=self.num_eval_episodes)

        # Store results
        self.results[algo_name] = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "training_time": training_time,
            "total_timesteps": self.num_timesteps,
        }

        # Save model
        model.save(f"{self.results_dir}/{algo_name}")

        env.close()
        eval_env.close()

        return model

    def train_mbrl(self, controller, dynamics_model=None):
        """Evaluate our MBRL approach."""
        print("\nEvaluating MBRL controller...")

        # Create evaluation environment
        eval_env = Monitor(gym.make(self.env_id))

        # Track metrics
        start_time = time.time()
        rewards = []

        # Run evaluation episodes
        for episode in range(self.num_eval_episodes):
            print(f"\nEpisode {episode + 1}/{self.num_eval_episodes}")
            total_reward = self.run_episode(eval_env, controller)
            rewards.append(total_reward)

        evaluation_time = time.time() - start_time

        # Calculate statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        # Store results
        self.results["MBRL"] = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "evaluation_time": evaluation_time,
            "num_episodes": self.num_eval_episodes,
        }

        eval_env.close()

    def run_episode(self, env, controller, max_steps=1000):
        """Run a single episode with the MBRL controller."""
        observation, _ = env.reset()
        total_reward = 0
        step = 0

        while step < max_steps:
            action, _ = controller.plan(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

            step += 1

        return total_reward

    def save_results(self):
        """Save benchmark results to file."""
        results_file = f"{self.results_dir}/benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=4)

        print("\nBenchmark Results:")
        print(json.dumps(self.results, indent=4))


def main():
    # Setup benchmark parameters
    env_id = "InvertedPendulum-v5"
    num_timesteps = 100000
    num_eval_episodes = 10

    # Initialize benchmark framework
    benchmark = BenchmarkFramework(env_id=env_id, num_timesteps=num_timesteps, num_eval_episodes=num_eval_episodes)

    # Train and evaluate PPO
    ppo_kwargs = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    }
    benchmark.train_model_free(PPO, "PPO", **ppo_kwargs)
    # Evaluate MBRL
    env = gym.make(env_id)
    checkpoint_path = "/Users/shuk/Desktop/mini_mbrl/mini_mbrl/03_cem_mpc/model_checkpoints/epoch_1.pth"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dynamics_model = load_model_from_checkpoint(checkpoint_path, state_dim, action_dim)

    controller = CEMMPCController(
        dynamics_model=dynamics_model,
        env=env,
        cem_horizon=10,
        num_simulated_trajectories=400,
        num_elites=40,
        optim_iterations=8,
    )
    benchmark.train_mbrl(controller)

    # Save results
    benchmark.save_results()


if __name__ == "__main__":
    main()
