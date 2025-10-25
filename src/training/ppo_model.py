import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class PPOActorCritic(nn.Module):
    """
    Combined Actor-Critic CNN for PPO.
    Outputs:
      - action distribution (Categorical)
      - value estimate (scalar)
    """

    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape

        # --- Shared feature extractor ---
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding_mode="circular"),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # --- Actor and Critic heads ---
        self.fc_shared = nn.Linear(conv_out_size, 512)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """Forward pass for both actor and critic."""
        x = x / 255.0
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        x = F.relu(self.fc_shared(features))

        logits = self.actor(x)
        value = self.critic(x)
        dist = Categorical(logits=logits)
        return dist, value
    
    def get_action_and_value(self, obs, action=None):
        """
        Returns action, log probability, entropy, and value estimate for PPO.
        If `action` is provided, it just evaluates that action (used for advantage calculation).
        """
        logits, value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, logprob, entropy, value
