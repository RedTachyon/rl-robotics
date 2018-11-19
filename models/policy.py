import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyFFN(nn.Module):

    def __init__(self, in_shape: int, out_shape: int):
        super(PolicyFFN, self).__init__()
        self.fc1 = nn.Linear(in_shape, 128)

        self.fc_mu = nn.Linear(128, out_shape)
        self.fc_var = nn.Linear(128, out_shape)

    def forward(self, x):
        x = F.elu(self.fc1(x))

        mu = self.fc_mu(x)
        var = self.fc_var(x)

        return mu, var
