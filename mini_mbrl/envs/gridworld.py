import copy
from typing import Any, Tuple, Dict, Optional
from utils import *

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def clamp(v, minimal_value, maximal_value):
    """Clamps a value between a minimum and maximum value to avoid the agent going out of bounds."""
    return min(max(v, minimal_value), maximal_value)


class GridWorldEnv(gym.Env):
    """
    This environment is a configurable variation on common grid world environments.
    Observation:
        Type: Box(2)
        Num     Observation            Min                     Max
        0       x position             0                       self.height - 1
        1       y position             0                       self.width - 1
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Go up
        1     Go right
        2     Go down
        3     Go left
        Each action moves a single cell.
    Reward:
        Reward is 0 at non-goal points, 1 at goal positions, and -1 at traps
    Starting State:
        The agent is placed at the start position specified during initialization
    Episode Termination:
        Agent has reached a goal or a trap.
    """

    def __init__(
        self,
        height=4,
        width=4,
        start_position=(0, 0),
        goal_positions=None,
        trap_positions=None,
    ):
        self.height = height
        self.width = width
        self.start_position = list(start_position)
        self.goal_positions = goal_positions or [
            (height - 1, 0),
            (height - 1, width - 1),
        ]
        self.trap_positions = trap_positions or [(height - 1, width // 2)]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.height - 1, self.width - 1]),
            dtype=np.int32,
        )

        self.agent_position = self.start_position.copy()
        self.done = False

        # Create the map
        self.map = [
            [
                "s" if (i, j) == tuple(self.start_position) else " "
                for j in range(self.width)
            ]
            for i in range(self.height)
        ]
        for i, j in self.goal_positions:
            self.map[i][j] = "g"
        for i, j in self.trap_positions:
            self.map[i][j] = "t"
        print(
            f"This is an environment with height {self.height} and width {self.width}"
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        self.agent_position = self.start_position.copy()
        return self._observe(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Updates the environment based on the action and returns the new observation, reward, done, and info."""
        reward = None
        done = None

        # Update the agent's position based on the action
        if action == 0:
            self.agent_position[0] = clamp(
                self.agent_position[0] - 1, 0, self.height - 1
            )
        elif action == 1:
            self.agent_position[1] = clamp(
                self.agent_position[1] + 1, 0, self.width - 1
            )
        elif action == 2:
            self.agent_position[0] = clamp(
                self.agent_position[0] + 1, 0, self.height - 1
            )
        elif action == 3:
            self.agent_position[1] = clamp(
                self.agent_position[1] - 1, 0, self.width - 1
            )
        else:
            raise ValueError("Invalid action")

        observation = self._observe()
        # Update the reward and done based on the new agent position
        if tuple(self.agent_position) in self.goal_positions:
            reward = 1
            done = True
            info = "Goal reached"
        elif tuple(self.agent_position) in self.trap_positions:
            reward = -1
            done = True
            info = "Trap reached"
        else:
            reward = -0.1
            done = False
            info = None

        return observation, reward, done, False, info

    def render(self):
        rendered_map = copy.deepcopy(self.map)
        rendered_map[self.agent_position[0]][self.agent_position[1]] = "A"
        print("-" * (self.width + 4))
        for row in rendered_map:
            print("".join(["|", " "] + row + [" ", "|"]))
        print("-" * (self.width + 4))
        return None

    def close(self):
        pass

    def _observe(self):
        return np.array(self.agent_position)
