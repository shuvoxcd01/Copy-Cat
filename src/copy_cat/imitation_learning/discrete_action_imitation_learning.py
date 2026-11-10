import logging

from tqdm import trange

from copy_cat.policies.base_policy import BasePolicy
from copy_cat.data.replay_buffer import SimpleReplayBuffer
from copy_cat.data.trajectory import Trajectory
from typing import List, Optional
import numpy as np
import torch


class DiscreteActionImitationLearning:
    def __init__(
        self,
        learner_policy: BasePolicy,
        expert_trajectories: List[Trajectory],
        optimizer: Optional[torch.optim.Optimizer] = None,
        stop_on_plateau: bool = True,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
    ):
        self.learner_policy = learner_policy
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

        # The frames are near-white (mean pixel ~0.99), so every conv1 filter sees a
        # large, almost constant drive. At lr=1e-3 Adam pushes their biases negative
        # within the first epoch and the whole layer stops firing; the net then emits a
        # constant logit vector and the loss parks at ln(2)=0.693 with zero gradient.
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.Adam(self.learner_policy.parameters(), lr=learning_rate)
        )
        self.stop_on_plateau = stop_on_plateau
        self.expert_trajectories = expert_trajectories
        self.replay_buffer = SimpleReplayBuffer()

        for trajectory in expert_trajectories:
            for i in range(trajectory.get_trajectory_length()):
                state, action, reward = trajectory.get_step(i)
                self.replay_buffer.store(state, action, reward, None, False, False)

        # A demonstration is mostly noop, so an unweighted loss is minimised by never
        # acting. Weighting each class by the inverse of its frequency makes a rare
        # keypress count as much as the many idle frames around it.
        labels = torch.tensor(
            [int(np.argmax(action)) for (_, action, *_) in self.replay_buffer.buffer]
        )
        counts = torch.bincount(
            labels, minlength=self.learner_policy.num_actions
        ).float()
        self.class_weight = counts.sum() / (len(counts) * counts)

    def train(self, num_epochs: int = 10):
        num_batches = len(self.replay_buffer) // self.batch_size

        for epoch in trange(num_epochs):
            epoch_loss = 0.0
            for _ in range(num_batches):
                observations, actions, rewards, next_states, terminated, truncated = self.replay_buffer.sample(
                    batch_size=self.batch_size, sequential=False
                )
                observations_tensor = torch.from_numpy(observations)
                expert_actions_tensor = torch.from_numpy(actions)

                self.optimizer.zero_grad()
                learner_action_logits = self.learner_policy(observations_tensor)
                loss = torch.nn.functional.cross_entropy(
                    learner_action_logits,
                    expert_actions_tensor,
                    weight=self.class_weight,
                )

                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                
            epoch_loss /= num_batches
            self.logger.debug(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        return self.learner_policy
