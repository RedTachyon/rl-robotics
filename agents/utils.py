from collections import namedtuple
from typing import Optional
import random

import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity: int):
        """
        Container for storing transitions encountered by an agent, with a predefined maximum size.

        Args:
            capacity: maximum amount of elements stored
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self):
        self.env = None
        self.current_state = None

    def select_action(self):
        raise NotImplementedError

    def take_action(self, action: Optional[int] = None, remember: bool = True, greedy: Optional[bool] = None):
        raise NotImplementedError

    def optimize_model(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def is_success(self, dist: float=.02) -> bool:
        """
        Checks whether the current state of the agent is successful
        Args:
            dist: error tolerance

        Returns:
            whether the state is considered successful

        """
        state = self.current_state.cpu().numpy().ravel()
        x_t, y_t = state[2], state[3]

        return np.linalg.norm([x_t, y_t]) < dist

    def is_reachable(self):
        """
        Checks whether it's possible to reach the target

        Returns:
            whether the target is reachable

        """
        state = self.current_state.cpu().numpy().ravel()
        return np.linalg.norm(state[:2]) <= 0.21


def describe_state(obs):
    target_x, target_y, to_t_x, to_t_y, costheta, sintheta, thetadot, gamma, gammadot = obs

    print('Target x: %.2f\nTarget y: %.2f\ntip x: %.2f\ntip y: %.2f\ntheta: %.2f\ngamma: %.2f'
          % (target_x, target_y, target_x + to_t_x, target_y + to_t_y, np.arccos(costheta), gamma))

    print('Length: %.3f' % np.linalg.norm(obs[:2] + obs[2:4]))
