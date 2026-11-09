import torch
from torch import nn
import torch.nn.functional as F

from copy_cat.policies.base_policy import BasePolicy
    

class BaseParameterizedPolicy(nn.Module, BasePolicy):
    def __init__(
        self,
        observation_shape: tuple,
        num_actions: int,
    ):
        nn.Module.__init__(self)
        BasePolicy.__init__(self)

        self.observation_shape = observation_shape
        self.num_actions = num_actions

    def get_action(self, state):
        action_probs = self.forward(state)

        action_probs = F.softmax(action_probs, dim=-1)

        action = torch.multinomial(action_probs, num_samples=1).detach().cpu().item()

        return action
    
    def get_actions(self, states):
        action_probs = self.forward(states)

        action_probs = F.softmax(action_probs, dim=-1)

        actions = torch.multinomial(action_probs, num_samples=1)

        return actions

    def get_action_prob(self, state, action):
        action_probs = self.forward(state)

        action_probs = F.softmax(action_probs, dim=-1)

        return action_probs[action]

    def get_log_action_prob(self, state: torch.Tensor, action: int) -> torch.Tensor:
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[action]

    def get_all_action_probabilities(self, states):
        action_probs = self.forward(states)

        action_probs = F.softmax(action_probs, dim=-1)

        return action_probs
