from typing import Optional, Dict, Tuple
from itertools import accumulate

import numpy as np

from gym.wrappers.time_limit import TimeLimit

import matplotlib.pyplot as plt
import seaborn as sns

import torch

from tqdm import tqdm

from .agents.utils import Agent
from .agents.DQN import DQNAgent
from .agents.VPG import VPGAgent

from .models.policy import PolicyFFN

sns.set()


def train_dqn_agent(env: TimeLimit, num_episodes: int = 5000, config: Optional[Dict] = None, device: str = 'cpu',
                    show: bool = True) -> Agent:
    """
    Trains a DQN agent on an environment for a specified number of episodes.

    Args:
        env: gym-compliant environment for the agent to operate in
        num_episodes: number of episodes the agent should learn on
        config: config dictionary for the agent
        device: device that is supposed to be used for model optimization
        show: whether plots should be displayed

    Returns:
        agent: a trained DQN agent

    """

    if config is None:
        config = dict()

    agent = DQNAgent(env, config, device)

    episode_scores = []
    episode_successes = []

    for i_episode in tqdm(range(num_episodes)):
        agent.reset()
        ep_score = 0
        ep_success = False

        while True:
            next_state, reward, done, info = agent.take_action()

            ep_score += reward
            if not ep_success:
                ep_success = agent.is_success()

            agent.optimize_model()

            if done:
                episode_scores.append(ep_score)
                episode_successes.append(int(ep_success))
                break

        if i_episode % agent.config['TARGET_UPDATE'] == 0:
            agent.update_target()

    episode_scores = torch.cat(episode_scores).cpu().numpy()

    if show:
        x_score = np.arange(len(episode_scores))
        y_score = episode_scores
        sns.regplot(x_score, y_score, lowess=True, marker='.')
        plt.show()

        x_success = np.arange(len(episode_successes))
        y_success = np.array(list(accumulate(episode_successes))) / (x_success + 1)

        sns.regplot(x_success, y_success, marker='.')
        plt.show()

    return agent


def train_vpg_agent(env: TimeLimit, num_episodes: int = 5000, config: Optional[Dict] = None, device: str = 'cpu',
                    show: bool = True) -> Agent:
    # TODO: implement this
    if config is None:
        config = dict()

    agent = VPGAgent(env, config, device)

    episode_scores = []
    episode_successes = []

    for i_episode in tqdm(range(num_episodes)):
        agent.reset()
        ep_score = 0
        episode_success = False
        while True:
            next_state, reward, done, info = agent.take_action()

            ep_score += reward
            if not episode_success:
                episode_success = agent.is_success()

            if done:
                episode_scores.append(ep_score)
                episode_successes.append(int(episode_success))
                break

        agent.optimize_model()

    episode_scores = torch.cat(episode_scores).cpu().numpy()

    if show:
        x_score = np.arange(len(episode_scores))
        y_score = episode_scores
        sns.regplot(x_score, y_score, lowess=True, marker='.')
        plt.show()

        x_success = np.arange(len(episode_successes))
        y_success = np.array(list(accumulate(episode_successes))) / (x_success + 1)

        sns.regplot(x_success, y_success, marker='.')
        plt.show()

    return agent


def evaluate_model(agent: Agent, num_episodes: int = 1000, show: bool = True) -> Tuple[float, float]:
    """
    Evaluates the agent on its environment for a specified number of episodes

    Args:
        agent:
        num_episodes:
        show:

    Returns:

    """
    episode_scores = []
    episode_successes = []

    for _ in tqdm(range(num_episodes)):
        agent.reset()

        ep_score = 0
        ep_success = False

        while True:
            agent.env.render()
            next_state, reward, done, info = agent.take_action(remember=False, greedy=True)

            ep_score += reward
            if not ep_success:
                ep_success = agent.is_success()

            if done:
                episode_scores.append(ep_score)
                episode_successes.append(int(ep_success))
                break

    test_episode_scores = torch.cat(episode_scores).cpu().numpy()

    if show:
        x_score = np.arange(len(episode_scores))
        y_score = episode_scores
        sns.regplot(x_score, y_score, lowess=True, marker='.')
        plt.show()

        x_success = np.arange(len(episode_successes))
        y_success = np.array(list(accumulate(episode_successes))) / (x_success + 1)

        sns.regplot(x_success, y_success, marker='.')
        plt.show()

    mean_score: float = test_episode_scores.mean()
    success_rate: float = np.mean(episode_successes)

    return mean_score, success_rate
