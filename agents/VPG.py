from collections import defaultdict
from typing import Type, Optional
import copy
import random
import math

from gym.wrappers import TimeLimit

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from ..models.policy import PolicyFFN
from .utils import Agent, ReplayMemory, Transition


class VPGAgent(Agent):

    def __init__(self, env: TimeLimit, config: dict, device: str = 'cpu', network: Type = PolicyFFN):
        super(VPGAgent, self).__init__()
        self.env = env
        self.device = device
        assert self.device in ['cpu', 'cuda'], ValueError("Device should be either 'cpu' or 'cuda'")

        self.type = torch.FloatTensor if self.device == 'cpu' else torch.cuda.FloatTensor

        self.current_state = self.reset()

        self._proto_config = config
        self.config = self.get_config()

        self.in_shape = env.observation_space.shape[0]
        self.out_shape = env.action_space.shape[0]
        self.policy = network(self.in_shape, self.out_shape).to(self.device).type(self.type)

    def select_action(self):
        with torch.no_grad():
            mu, var = self.policy(self.current_state)
        var = F.softplus(var)

        eps = torch.randn(mu.size()).to(self.device).type(self.type)
        action = mu + torch.sqrt(var)*eps

        return action

    def take_action(self, action: Optional[int] = None, remember: bool = True):
        pass

    def optimize_model(self, rewards=None, histories=None):
        pass

    def reset(self):
        """
        Resets the environment and updates the current state.

        Returns:
            observation
        """
        self.current_state = torch.tensor([self.env.reset()], device=self.device).type(self.type)
        return self.current_state

    def is_success(self) -> bool:
        pass

    def _add_to_config(self, name, default):
        """
         Helper method for building the config dictionary.
        """
        if name in self._proto_config.keys():
            self.config[name] = self._proto_config[name]
        else:
            self.config[name] = default

    def get_config(self):
        """
        Builds the config dictionary using the proto config provided in the constructor.

        Returns: config, dict: keeps relevant hyperparameters for the agent

        """
        self.config = {}

        # Add necessary default parameters
        self._add_to_config('BATCH_SIZE', 32)
        self._add_to_config('GAMMA', 0.99)
        self._add_to_config('EPS_START', 0.9)
        self._add_to_config('EPS_END', 0.05)
        self._add_to_config('EPS_DECAY', 200)
        self._add_to_config('TARGET_UPDATE', 10)
        self._add_to_config('MEMORY_SIZE', 10000)
        self._add_to_config('ACTION_DICT', {
            0: [.0, -.1],
            1: [.0, .1],
            2: [.1, -.1],
            3: [.1, .0],
            4: [.1, .1],
            5: [-.1, -.1],
            6: [-.1, .0],
            7: [-.1, .1],
            8: [.0, .0]
        })

        # Add everything else provided
        for key in self._proto_config.keys():
            if key not in self.config:
                self.config[key] = self._proto_config[key]

        return self.config