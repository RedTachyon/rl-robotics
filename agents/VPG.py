from collections import defaultdict
from typing import Type, Optional, Tuple
import copy
import random
import math

from gym.wrappers import TimeLimit

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
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

        self.rewards = []
        self.log_probs = []

        self.optimizer = optim.Adam(self.policy.parameters())

    def select_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # with torch.no_grad():
        mu, var = self.policy(self.current_state)

        var = F.softplus(var).flatten()

        dist = MultivariateNormal(loc=mu, covariance_matrix=torch.diag(var))

        action = dist.sample().flatten()
        logprob = dist.log_prob(action)

        return action, logprob

    def take_action(self, action=None, remember=True, greedy=None):
        """
        Acts upon the environment with an indicated or self-chosen action, stores the transition in memory and updates
        the state of the environment.

        Args:
            action: None or list[float]: if None, action is selected automatically;
                    else, the preferred action
            remember: bool, whether or not to store the transition
            greedy: unused

        Returns:
            new_state: observation, stored also in self.current_state
            reward: float, reward for this transition
            done: bool, whether the episode is finished
            info: any

        """
        logprob = None
        if action is None:
            action, logprob = self.select_action()

        new_state, reward, done, info = self.env.step(action.cpu().numpy())

        new_state = torch.tensor([new_state], device=self.device).type(self.type)
        reward = torch.tensor([reward], device=self.device).type(self.type)

        if remember and logprob is not None:
            self.rewards.append(reward)
            self.log_probs.append(logprob)

        self.current_state = new_state

        return new_state, reward, done, info

    def optimize_model(self):
        GAMMA: float = self.config['GAMMA']

        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + GAMMA * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards)
        # Modify rewards to advantage or something
        # rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        for log_prob, reward in zip(self.log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()

        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()

        self.optimizer.step()

        del self.rewards[:]
        del self.log_probs[:]

    def reset(self):
        """
        Resets the environment and updates the current state.

        Returns:
            observation
        """
        self.current_state = torch.tensor([self.env.reset()], device=self.device).type(self.type)
        return self.current_state

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
        self._add_to_config('NUM_TRAJECTORIES', 32)
        self._add_to_config('GAMMA', 0.8)

        # Add everything else provided
        for key in self._proto_config.keys():
            if key not in self.config:
                self.config[key] = self._proto_config[key]

        return self.config
