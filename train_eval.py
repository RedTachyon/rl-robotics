from typing import Optional, Dict, Type, Tuple

import numpy as np

from gym.wrappers.time_limit import TimeLimit

import matplotlib.pyplot as plt
import seaborn as sns

import torch

from tqdm import tqdm

from agents import DQNAgent, Agent


sns.set()


def train_dqn_agent(env: TimeLimit, num_episodes: int=5000, config: Optional[Dict]=None, device: str='cpu') -> Agent:
    """
    Trains a DQN agent on an environment for a specified number of episodes.

    Args:
        env: gym-compliant environment for the agent to operate in
        num_episodes: number of episodes the agent should learn on
        config: config dictionary for the agent
        device: device that is supposed to be used for model optimization

    Returns:
        agent: a trained DQN agent

    """

    if config is None:
        config = dict()

    agent = DQNAgent(env, config, device)

    episode_scores = []

    for i_episode in tqdm(range(num_episodes)):
        agent.reset()
        ep_score = 0

        while True:
            next_state, reward, done, info = agent.take_action()

            ep_score += reward

            agent.optimize_model()

            if done:
                episode_scores.append(ep_score)
                break

        if i_episode % agent.config['TARGET_UPDATE'] == 0:
            agent.update_target()

    episode_scores = torch.cat(episode_scores).cpu().numpy()

    sns.regplot(np.arange(len(episode_scores)), episode_scores, lowess=True, marker='.')

    plt.show()

    return agent


def evaluate_model(agent: Agent, num_episodes: int=1000) -> float:

    test_episode_scores = []

    for i_episode in tqdm(range(num_episodes)):
        agent.reset()

        ep_score = 0
        while True:
            agent.env.render()
            next_state, reward, done, info = agent.take_action(remember=False, greedy=True)

            ep_score += reward

            if done:
                test_episode_scores.append(ep_score)
                break

    test_episode_scores = torch.cat(test_episode_scores).cpu().numpy()

    sns.regplot(np.arange(len(test_episode_scores)), test_episode_scores, marker='.')
    plt.show()

    mean_score = test_episode_scores.mean()

    return mean_score

def save_agent(agent, path):
    to_save = (agent.__class__, agent.policy_net)

def load_agent(path: str) -> Agent:
    pass # TODO: implement this
