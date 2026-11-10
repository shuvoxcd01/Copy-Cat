
import torch

from copy_cat.imitation_learning.discrete_action_imitation_learning import DiscreteActionImitationLearning


from copy_cat.policies.parameterized.discrete_action_cnn_policy import DiscreteActionCNNPolicy
from copy_cat.storage.trajectory_store import TrajectoryStore
from copy_cat.preprocessing.action_preprocessor import (
    num_actions_from_names,
)
import logging

# Configure logging to display debug messages
logging.basicConfig(level=logging.DEBUG)

store = TrajectoryStore("./recordings/TRexRunner")
selected_episodes = store.list()

trajectories = []

for selected_episode in selected_episodes:
    trajectory = store.load(selected_episode)
    trajectories.append(trajectory)

observation_shape = trajectory.get_step(0)[0].shape
num_actions = num_actions_from_names(trajectory.action_names)

learner_policy = DiscreteActionCNNPolicy(observation_shape=observation_shape, num_actions=num_actions)

algorithm = DiscreteActionImitationLearning(
    expert_trajectories=trajectories, learner_policy=learner_policy, optimizer=None,
)

trained_policy = algorithm.train(num_epochs=200)
trained_policy.save("./trained_policy_6.pt")

print("Trained policy:", trained_policy)

