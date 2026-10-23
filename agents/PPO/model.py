"""Gaussian latent policy with a bijective map into the unit action disk."""
import math
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(0.5)))
        self.actor = nn.Sequential(nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, action_dim))
        self.critic = nn.Sequential(nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1))
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def distribution(self, state):
        mean, value = self(state)
        return Normal(mean, self.log_std.clamp(math.log(.05), math.log(1.)).exp()), value

    @staticmethod
    def squash(latent):
        return latent / torch.sqrt(1 + latent.square().sum(dim=-1, keepdim=True))

    def log_prob(self, distribution, latent):
        # For a=z/sqrt(1+z'z), det(da/dz)=(1+z'z)^(-(d+2)/2).
        return distribution.log_prob(latent).sum(-1) + (self.action_dim + 2) / 2 * torch.log1p(latent.square().sum(-1))
