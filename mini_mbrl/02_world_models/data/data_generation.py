# This script generates data for the CarRacing-v3 environment using a random policy(white noise or brownian motion). Pink noise will be added later.

import gymnasium as gym
import numpy as np
import argparse
import math
import os
import cv2
import multiprocessing
from functools import partial


def sample_continuous_policy(action_space, seq_len, dt):
    """Sample a continuous policy as Brownian motion."""
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(
                actions[-1] + math.sqrt(dt) * daction_dt,
                action_space.low,
                action_space.high,
            )
        )
    return actions


def preprocess_frame(frame):
    """Crop and resize the frame to 64x64 to remove game status and black lines."""
    cropped_frame = frame[:-80, :-80, :]
    resized_frame = cv2.resize(cropped_frame, (64, 64), interpolation=cv2.INTER_AREA)
    return resized_frame


def generate_single_rollout(i, seq_len, data_dir, noise_type):
    """Generates a single rollout and saves it."""
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    state, _ = env.reset()

    if noise_type == "white":
        a_rollout = [env.action_space.sample() for _ in range(seq_len)]
    elif noise_type == "brown":
        a_rollout = sample_continuous_policy(env.action_space, seq_len, 1.0 / 50)

    s_rollout = []
    a_rollout_recorded = []

    t = 0
    while t < seq_len:
        action = a_rollout[t]
        t += 1

        # Step the environment and capture RGB observation
        state, reward, done, truncated, _ = env.step(action)
        frame = env.render()  # Get the RGB frame

        processed_frame = preprocess_frame(frame)

        s_rollout.append(processed_frame)  # Save the processed frame
        a_rollout_recorded.append(action)  # Save the action

        if done or truncated:
            break

    print(f"> End of rollout {i}, {len(s_rollout)} frames...")
    np.savez_compressed(
        os.path.join(data_dir, f"rollout_{i}.npz"),
        observations=np.array(s_rollout),
        actions=np.array(a_rollout_recorded),
    )
    env.close()


def generate_data(rollouts, data_dir, noise_type, num_processes):
    """Generates data using multiprocessing."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Define the length of each rollout
    seq_len = 1000

    # Use multiprocessing to run the rollouts in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(
            partial(
                generate_single_rollout,
                seq_len=seq_len,
                data_dir=data_dir,
                noise_type=noise_type,
            ),
            range(rollouts),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, help="Number of rollouts", default=100)
    parser.add_argument(
        "--dir", type=str, help="Where to place rollouts", default="dataset"
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["white", "brown"],
        help="Noise type used for action sampling.",
        default="brown",
    )
    parser.add_argument(
        "--processes", type=int, help="Number of processes for parallelism", default=8
    )
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy, args.processes)
    # generate_single_rollout(0, 1000, 'dataset', 'brown')
