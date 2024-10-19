import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse
from os.path import join


def visualize_rollout(file_path, save_as_gif=False):
    # Load the recorded rollout
    data = np.load(file_path)
    frames = data["observations"]
    actions = data["actions"]

    # Set up the matplotlib figure
    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])  # Show the first frame
    plt.axis("off")

    def update_frame(t):
        """Updates the frame shown by the plot"""
        img.set_data(frames[t])
        ax.set_title(f"Step: {t}, Action: {actions[t]}")
        return [img]

    # If you want to save as a GIF
    if save_as_gif:
        gif_frames = [frame for frame in frames]
        gif_path = file_path.replace(".npz", ".gif")
        imageio.mimsave(gif_path, gif_frames, fps=30)
        print(f"GIF saved at: {gif_path}")
    else:
        # Display the animation using matplotlib
        from matplotlib.animation import FuncAnimation

        anim = FuncAnimation(fig, update_frame, frames=len(frames), interval=50)
        plt.show()


def visualize_entire_rollout(data_dir, save_as_gif=False):
    """Apply the visualize_rollout function to all rollouts in a directory"""
    import os

    for file in os.listdir(data_dir):
        if file.endswith(".npz"):
            file_path = join(data_dir, file)
            visualize_rollout(file_path, save_as_gif=save_as_gif)


def visualize_entire_rollout_multiprocess(data_dir, save_as_gif=False):
    """Apply the visualize_rollout function to all rollouts in a directory using multiprocessing"""
    import os
    from functools import partial
    from multiprocessing import Pool

    visualize_rollout_partial = partial(visualize_rollout, save_as_gif=save_as_gif)

    with Pool() as p:
        p.map(
            visualize_rollout_partial,
            [
                join(data_dir, file)
                for file in os.listdir(data_dir)
                if file.endswith(".npz")
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, help="Path to the .npz file folder", required=True
    )
    parser.add_argument(
        "--save_gif", action="store_true", help="Save the rollout as a GIF"
    )
    args = parser.parse_args()

    # visualize_rollout(args.file, save_as_gif=args.save_gif)
    # visualize_entire_rollout(args.dir, save_as_gif=args.save_gif)
    visualize_entire_rollout_multiprocess(args.dir, save_as_gif=args.save_gif)
