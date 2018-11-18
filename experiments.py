from typing import Dict, Optional, Tuple

import pybullet_envs as pe

from . import train_eval as te
from .agents.utils import Agent


def train_eval_reacher_dqn(num_train: int,
                           num_test: int,
                           device: str,
                           config: Optional[Dict] = None,
                           viz: bool = False) -> Tuple[Agent, float, float]:
    assert device in ('cpu', 'cuda'), "The device must be either cpu or cuda"

    env = pe.make('ReacherBulletEnv-v0')
    print('Training the agent.')
    agent = te.train_dqn_agent(env, num_train, config, device, show=True, viz=viz)
    print()
    print('\nTraining finished. Evaluating performance.')

    mean_score, success_rate = te.evaluate_model(agent, num_test, show=True)

    return agent, mean_score, success_rate


# if __name__ == '__main__':
#     config = {
#         'TARGET_UPDATE': 10,
#         'BATCH_SIZE': 32,
#         'MEMORY_SIZE': 30000,
#         'EPS_START': 0.9,
#         'EPS_END': 0.01,
#         'EPS_DECAY': 500,
#         'GAMMA': 0.8,
#
#     }
#
#     agent, mean_score, success_rate = train_eval_reacher_dqn(100, 100, 'cpu')
#     print("Score: %.3f\nSuccess rate: %.3f" % (mean_score, success_rate))
