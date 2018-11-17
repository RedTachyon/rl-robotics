import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):

    def __init__(self, in_shape: int, out_shape: int):
        """
        A regular feed-forward, elu-activated neural network to use for the Deep Q Learning algorithm.

        Args:
            in_shape:
            out_shape:
        """
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_shape, 128)
        self.fc2 = nn.Linear(128, 128)

        self.head = nn.Linear(128, out_shape)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.head(x)
