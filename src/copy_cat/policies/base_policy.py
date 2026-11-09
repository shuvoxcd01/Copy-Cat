from abc import ABC, abstractmethod
import logging
import os

import torch


class BasePolicy(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def get_action_prob(self, state, action):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def get_all_action_probabilities(self, states):
        raise NotImplementedError("This method must be overridden")

    @abstractmethod
    def update(self, state, action):
        raise NotImplementedError("This method must be overridden")

    def __call__(self, state, *args, **kwds):
        return self.get_action(state=state)

    def save(self, path: str):
        """
        Save the policy to a file.

        Args:
            path: The file path where the policy should be saved.

        Returns:
            None.
        """
        path = str(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(self, path)

    @staticmethod
    def load(path: str):
        """
        Load the policy from a file.

        Args:
            path: The file path from which the policy should be loaded.

        Returns:
            An instance of the policy.
        """
        policy = torch.load(str(path), map_location="cpu")

        if not isinstance(policy, BasePolicy):
            raise TypeError(f"File '{path}' does not contain a BasePolicy instance.")

        return policy
