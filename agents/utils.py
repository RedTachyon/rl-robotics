from collections import namedtuple
from typing import Optional
import random

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

    def select_action(self, greedy: bool = False):
        raise NotImplementedError

    def take_action(self, action: Optional[int] = None, remember: bool = True, greedy: bool = False):
        raise NotImplementedError

    def optimize_model(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def is_success(self) -> bool:
        raise NotImplementedError
