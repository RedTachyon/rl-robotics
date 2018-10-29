from collections import namedtuple, defaultdict
from itertools import count
import random
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pybullet as p
import pybullet_envs as pe

import gym

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')
print(device)
float_type = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, in_shape, out_shape):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_shape, 128)
        self.fc2 = nn.Linear(128, 128)

        self.head = nn.Linear(128, out_shape)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.head(x)


class DQNAgent:

    def __init__(self, env, in_shape: int, out_shape: int, config: dict, type_=torch.cuda.FloatTensor):
        self.env = env
        self.type = type_
        self.current_state = self.reset()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._proto_config = config
        self.config = self.get_config()

        self.policy_net = DQN(in_shape, out_shape).to(device).type(self.type)
        self.target_net = DQN(in_shape, out_shape).to(device).type(self.type)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(self.config['MEMORY_SIZE'])

        self.histories = defaultdict(list)

        self.steps_done = 0

    def select_action(self, greedy=False):
        if greedy:
            with torch.no_grad():
                return self.policy_net(self.current_state).max(1)[1].view(1, 1).item()

        EPS_END, EPS_START, EPS_DECAY = self.config['EPS_END'], self.config['EPS_START'], self.config['EPS_DECAY']
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():

                return self.policy_net(self.current_state).max(1)[1].view(1, 1).item()
        else:
            return random.choice(list(self.config['ACTION_DICT'].keys()))

    def add_to_config(self, name, default):
        if name in self._proto_config.keys():
            self.config[name] = self._proto_config[name]
        else:
            self.config[name] = default

    def get_config(self):
        self.config = {}

        # Add necessary parameters
        self.add_to_config('BATCH_SIZE', 128)
        self.add_to_config('GAMMA', 0.99)
        self.add_to_config('EPS_START', 0.9)
        self.add_to_config('EPS_END', 0.05)
        self.add_to_config('EPS_DECAY', 200)
        self.add_to_config('TARGET_UPDATE', 10)
        self.add_to_config('MEMORY_SIZE', 10000)
        self.add_to_config('ACTION_DICT', {
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

    def take_action(self, action=None, remember=True, greedy=False):
        if action is None:
            action = self.select_action(greedy=greedy)

        new_state, reward, done, info = self.env.step(self.config['ACTION_DICT'][action])

        #         if done:
        #             reward = -1

        new_state = torch.tensor([new_state], device=device).type(self.type)
        reward = torch.tensor([reward], device=device).type(self.type)

        if remember:
            self.memory.push(self.current_state, action, new_state, reward)

        self.current_state = new_state

        return new_state, reward, done, info

    def optimize_model(self):
        BATCH_SIZE, GAMMA = self.config['BATCH_SIZE'], self.config['GAMMA']

        if len(self.memory) < BATCH_SIZE:
            return False

        self.policy_net.train()

        # Randomly sample and preprocess a batch of data
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        #
        state_batch = torch.cat(batch.state).type(self.type)
        action_batch = torch.tensor(batch.action, device=device).view(-1, 1)
        reward_batch = torch.cat(batch.reward).type(self.type)
        next_state_batch = torch.cat(batch.next_state).type(self.type)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # Estimate from the network

        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Estimate from Bellman equation and the network

        # Basically training to solve Bellman equation

        loss = F.smooth_l1_loss(state_action_values.view(-1, 1), expected_state_action_values.view(-1, 1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        self.policy_net.eval()

        return True

    def reset(self):
        self.current_state = torch.tensor([self.env.reset()], device=device).type(self.type)
        return self.current_state

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())