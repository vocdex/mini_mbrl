import gym
import numpy as np


class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, *args, **kwargs):
        return self.env.action_space.sample()


class OpenLoopPolicy:
    """This should return an action when called with an observation."""

    def __init__(self, action_sequence, env: gym.Env):
        self.action_sequence = action_sequence
        self.env = env

    def get_action(self, obs, state, **kwargs):
        return self.action_sequence[0]


def sample_trajectory(env, policy, n_steps, is_deterministic: bool, seed: int) -> [str, np.ndarray]:
    """Samples a single trajectory using the given policy."""
    if is_deterministic:
        ob, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
    else:
        ob, _ = env.reset()
    (
        obs,
        next_obs,
        acs,
        rewards,
        dones,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )
    step = 0
    while True and step < n_steps:
        obs.append(ob)
        action = policy(ob)
        acs.append(action)
        ob, reward, done, _, _ = env.step(action)
        next_obs.append(ob)
        rewards.append(reward)
        dones.append(done)
        if done:
            break
        step += 1

    env.close()
    traj_stats = {"total_reward": np.sum(rewards), "length": step}
    return {
        "observations": np.array(obs),
        "next_observations": np.array(next_obs),
        "actions": np.array(acs),
        "rewards": np.array(rewards),
        "dones": np.array(dones),
        "traj_stats": traj_stats,
    }


def sample_n_trajectories(
    env: gym.Env, policy, num_traj: int, n_steps: int, is_deterministic: bool
) -> [str, np.ndarray]:
    """Samples a batch of trajectories using the given policy."""
    trajs = []
    # use different seeds for each trajectory
    for i in range(num_traj):
        traj = sample_trajectory(env, policy, n_steps, is_deterministic, seed=i)
        trajs.append(traj)
    return trajs


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    policy = RandomPolicy(env)
    trajs = sample_n_trajectories(env, policy, 10, 100, False)

    for traj in trajs:
        print("Length of trajectory: ", len(traj["observations"]))
        print("Total reward: ", traj["traj_stats"]["total_reward"])
