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
        raise NotImplementedError

    def get_action(self, obs, state, **kwargs):
        raise NotImplementedError


