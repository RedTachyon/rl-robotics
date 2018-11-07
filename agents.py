from collections import namedtuple, defaultdict
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# float_type = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

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

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, in_shape: int, out_shape: int):
        """
        A regular feed-forward, elu-activated neural network to use for the Deep Q Learning algorithm.

        Args:
            in_shape:
            out_shape:
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_shape, 128)
        self.fc2 = nn.Linear(128, 128)

        self.head = nn.Linear(128, out_shape)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.head(x)


class Agent:
    def __init__(self):
        pass

    def select_action(self, greedy=False):
        raise NotImplementedError

    def take_action(self, action=None):
        raise NotImplementedError

    def optimize_model(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class DQNAgent(Agent):
    # TODO: add .cpu() .cuda() methods
    # TODO: integrate the success rate history
    def __init__(self, env, config: dict, device='cpu'):
        """
        Basic agent for interacting with an environment, using a Deep Q Learning algorithm with discrete actions.

        Args:
            env: environment in which the agent will operate
            config: dictionary containing relevant parameters for the agent;
            device: indicates on what device the agent should run; either 'cpu' or 'cuda'
        """
        super(DQNAgent, self).__init__()

        self.env = env
        self.device = device
        assert self.device in ['cpu', 'cuda'], ValueError("Device should be either 'cpu' or 'cuda'")

        self.type = torch.FloatTensor if self.device == 'cpu' else torch.cuda.FloatTensor

        self.current_state = self.reset()

        self._proto_config = config
        self.config = self.get_config()

        self.in_shape = self.env.observation_space.shape[0]
        self.out_shape = len(self.config['ACTION_DICT'])

        self.policy_net = DQN(self.in_shape, self.out_shape).to(self.device).type(self.type)
        self.target_net = DQN(self.in_shape, self.out_shape).to(self.device).type(self.type)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(self.config['MEMORY_SIZE'])

        self.histories = defaultdict(list)

        self.steps_done = 0

    def select_action(self, greedy=False):
        """
        Chooses an action to be taken in the current state of the agent's environment.
        Two strategies are supported: epsilon greedy (default, with greedy=False) or greedy (with greedy=True).
        If the epsilon-greedy strategy is used, the epsilon value is updated.
        Args:
            greedy: bool, whether or not to use a greedy strategy

        Returns:
            action: int, index of the discretized action that is suggested to be taken

        """
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

        # Add necessary parameters
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

    def take_action(self, action=None, remember=True, greedy=False):
        """
        Acts upon the environment with an indicated or self-chosen action, stores the transition in memory and updates
        the state of the environment.

        Args:
            action: None or int: if None, action is selected automatically; else, the index of the preferred action
            remember: bool, whether or not to store the transition
            greedy: bool, if action is None, indicates what strategy to use

        Returns:
            new_state: observation, stored also in self.current_state
            reward: float, reward for this transition
            done: bool, whether the episode is finished
            info: any

        """
        if action is None:
            action = self.select_action(greedy=greedy)

        new_state, reward, done, info = self.env.step(self.config['ACTION_DICT'][action])

        new_state = torch.tensor([new_state], device=self.device).type(self.type)
        reward = torch.tensor([reward], device=self.device).type(self.type)

        if remember:
            self.memory.push(self.current_state, action, new_state, reward)

        self.current_state = new_state

        return new_state, reward, done, info

    def optimize_model(self):
        """
        If there are enough transitions in the memory, perform a single DQ learning update using
        randomly sampled transitions.

        Returns:
        bool: whether the update was performed
        """
        BATCH_SIZE, GAMMA = self.config['BATCH_SIZE'], self.config['GAMMA']

        if len(self.memory) < BATCH_SIZE:
            return False

        self.policy_net.train()

        # Randomly sample and preprocess a batch of data
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        #
        state_batch = torch.cat(batch.state).type(self.type)
        action_batch = torch.tensor(batch.action, device=self.device).view(-1, 1)
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
        """
        Resets the environment and updates the current state.
        Returns:
        observation
        """
        self.current_state = torch.tensor([self.env.reset()], device=self.device).type(self.type)
        return self.current_state

    def update_target(self):
        """
        Updates the target network with the policy network's weights
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def is_success(self, dist=.1):
        x_t, y_t = self.current_state[2], self.current_state[3]

        return np.linalg.norm([x_t, y_t]) < dist
