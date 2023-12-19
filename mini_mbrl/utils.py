import gym
import numpy as np


class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, *args, **kwargs):
        return self.env.action_space.sample()


def sample_trajectory(env, policy, max_length, is_deterministic: bool, seed: int) -> [str, np.ndarray]:
    """Samples a single trajectory using the given policy."""
    if is_deterministic:
        ob, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
    else:
        ob, _ = env.reset(None)
    (
        obs,
        acs,
        rewards,
        next_obs,
        dones,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )
    step = 0
    while True and step < max_length:
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
    traj_stats = {"total_reward": np.sum(rewards), "length": len(rewards)}
    return {
        "observations": np.array(obs),
        "actions": np.array(acs),
        "rewards": np.array(rewards),
        "next_observations": np.array(next_obs),
        "dones": np.array(dones),
        "stats": traj_stats,
    }


def sample_trajectories(
    env: gym.Env, policy, num_traj: int, max_length: int, is_deterministic: bool
) -> [str, np.ndarray]:
    """Samples a batch of trajectories using the given policy."""
    trajs = []
    # use different seeds for each trajectory
    for i in range(num_traj):
        traj = sample_trajectory(env, policy, max_length, is_deterministic, seed=i)
        trajs.append(traj)
    return trajs


# if __name__ == "__main__":
#     env = gym.make("CartPole-v1")
#     policy = RandomPolicy(env)
#     trajs = sample_trajectories(env, policy, 2, 100, True)

#     for traj in trajs:
#         obs = traj["observations"]
#         mean_obs = np.mean(obs, axis=0)
#         print("Mean observation: ", mean_obs)
