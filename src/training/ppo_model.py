import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        C, H, W = input_shape

        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4, padding=0),  # adjust padding if needed
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)
            conv_out_size = self.conv(dummy_input).shape[1]

        # Actor head
        self.actor = nn.Linear(conv_out_size, n_actions)
        # Critic head
        self.critic = nn.Linear(conv_out_size, 1)

    def forward(self, x):
        """
        Forward pass returning logits and value.
        x: tensor (batch_size, C, H, W)
        """
        features = self.conv(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, action=None):
        """
        For rollout: sample action from policy, return logprob and value.
        x: tensor (batch_size, C, H, W)
        action: optional tensor of actions (used during evaluation)
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    def get_value(self, x):
        """
        Return value only (for GAE computation)
        """
        _, value = self.forward(x)
        return value

    def evaluate_actions(self, x, actions):
        """
        For PPO update: logprobs, entropy, and value for given actions
        x: tensor (batch_size, C, H, W)
        actions: tensor of actions (batch_size,)
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, entropy, value
