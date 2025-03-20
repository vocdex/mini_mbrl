""" FINALLY SOME CODE THAT WORKS! """

import time

import gymnasium as gym
from gymnasium import Wrapper


class StatefulMujocoWrapper(Wrapper):
    """
    A wrapper that adds state saving/loading capabilities to MuJoCo environments.
    """

    def __init__(self, env):
        super().__init__(env)
        self.initial_state = None
        # Store the initial random state when environment is created
        self._save_initial_state()

    def _save_initial_state(self):
        """Save the initial state after environment creation."""
        self.initial_state = self.save_state()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Update initial state after reset
        self._save_initial_state()
        return obs, info

    def save_state(self):
        """
        Save the current state of the environment.
        """
        state = {
            # MuJoCo simulation state
            "qpos": self.env.unwrapped.data.qpos.copy(),
            "qvel": self.env.unwrapped.data.qvel.copy(),
            "mocap_pos": self.env.unwrapped.data.mocap_pos.copy()
            if hasattr(self.env.unwrapped.data, "mocap_pos")
            else None,
            "mocap_quat": self.env.unwrapped.data.mocap_quat.copy()
            if hasattr(self.env.unwrapped.data, "mocap_quat")
            else None,
            # Environment-specific variables
            "step_count": getattr(self.env.unwrapped, "_step_count", 0),
            "elapsed_steps": getattr(self.env.unwrapped, "_elapsed_steps", 0),
            # Random state - updated for new NumPy Generator
            "np_random_state": self.env.unwrapped.np_random.bit_generator.state
            if hasattr(self.env.unwrapped, "np_random")
            else None,
        }
        return state

    def get_current_state(env):
        """
        Retrieve the current state of a MuJoCo-based environment.

        Parameters:
            env: The Gymnasium environment instance.

        Returns:
            A dictionary containing the current state.
        """
        current_state = {
            "qpos": env.unwrapped.data.qpos.copy(),  # Generalized positions
            "qvel": env.unwrapped.data.qvel.copy(),  # Generalized velocities
            "mocap_pos": env.unwrapped.data.mocap_pos.copy() if hasattr(env.unwrapped.data, "mocap_pos") else None,
            "mocap_quat": env.unwrapped.data.mocap_quat.copy() if hasattr(env.unwrapped.data, "mocap_quat") else None,
            "step_count": getattr(env.unwrapped, "_step_count", 0),  # Step count if available
            "elapsed_steps": getattr(env.unwrapped, "_elapsed_steps", 0),  # Elapsed steps if available
            "np_random_state": env.unwrapped.np_random.bit_generator.state
            if hasattr(env.unwrapped, "np_random")
            else None,  # Random generator state
        }
        return current_state

    def set_state(env, state):
        """
        Set the state of a MuJoCo-based environment.

        Parameters:
            env: The Gymnasium environment instance.
            state: A dictionary containing the state to set.
        """
        # Restore MuJoCo simulation state
        env.unwrapped.data.qpos[:] = state["qpos"]
        env.unwrapped.data.qvel[:] = state["qvel"]

        if state["mocap_pos"] is not None and hasattr(env.unwrapped.data, "mocap_pos"):
            env.unwrapped.data.mocap_pos[:] = state["mocap_pos"]

        if state["mocap_quat"] is not None and hasattr(env.unwrapped.data, "mocap_quat"):
            env.unwrapped.data.mocap_quat[:] = state["mocap_quat"]

        # Restore environment-specific variables
        if hasattr(env.unwrapped, "_step_count"):
            env.unwrapped._step_count = state["step_count"]
        if hasattr(env.unwrapped, "_elapsed_steps"):
            env.unwrapped._elapsed_steps = state["elapsed_steps"]

        # Restore random state
        if state["np_random_state"] is not None and hasattr(env.unwrapped, "np_random"):
            env.unwrapped.np_random.bit_generator.state = state["np_random_state"]

        # Forward the simulation using the MuJoCo API
        from mujoco import mj_forward

        mj_forward(env.unwrapped.model, env.unwrapped.data)

    def restore_state(self, state):
        """
        Restore the environment to a saved state.
        """
        # Restore MuJoCo simulation state
        self.env.unwrapped.data.qpos[:] = state["qpos"]
        self.env.unwrapped.data.qvel[:] = state["qvel"]

        if state["mocap_pos"] is not None and hasattr(self.env.unwrapped.data, "mocap_pos"):
            self.env.unwrapped.data.mocap_pos[:] = state["mocap_pos"]

        if state["mocap_quat"] is not None and hasattr(self.env.unwrapped.data, "mocap_quat"):
            self.env.unwrapped.data.mocap_quat[:] = state["mocap_quat"]

        # Restore environment-specific variables
        if hasattr(self.env.unwrapped, "_step_count"):
            self.env.unwrapped._step_count = state["step_count"]
        if hasattr(self.env.unwrapped, "_elapsed_steps"):
            self.env.unwrapped._elapsed_steps = state["elapsed_steps"]

        # Restore random state
        if state["np_random_state"] is not None and hasattr(self.env.unwrapped, "np_random"):
            self.env.unwrapped.np_random.bit_generator.state = state["np_random_state"]

        # Forward the simulation using the MuJoCo API
        from mujoco import mj_forward

        mj_forward(self.env.unwrapped.model, self.env.unwrapped.data)

    def reset_to_initial_state(self):
        """
        Reset the environment to its initial state after the last reset.
        """
        if self.initial_state is not None:
            self.restore_state(self.initial_state)
            return self.env.unwrapped._get_obs()
        else:
            return self.reset()


if __name__ == "__main__":
    # Create environment with the wrapper
    env = gym.make("InvertedPendulum-v5", render_mode="human")
    env = StatefulMujocoWrapper(env)

    # Reset and get initial observation
    obs, _ = env.reset()

    # Save current state
    saved_state = env.save_state()
    print(saved_state)
    middle_state = None
    # Take some actions
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("Step count:", _)
        if _ == 100:
            middle_state = env.save_state()
            print("Saved state at step 100")
        env.render()

    # Restore to saved state
    env.restore_state(middle_state)
    env.render()
    print("Restored to step 100")
    current_state = env.get_current_state()
    print("Current state after restore:", current_state)
    print("Middle state:", middle_state)

    time.sleep(3)
    # Restore to initial state
    env.reset_to_initial_state()
    env.render()
    print("Restored to initial state")
    current_state = env.get_current_state()
    print("Current state after restore:", current_state)
    print("Initial state:", env.initial_state)

    # Sleep for a while to see the restored state
    time.sleep(3)
    # Close the environment
    env.close()
