"""Train a CNN policy by behaviour cloning on every recorded TRex Runner episode.

    python examples/trex_runner/train_policy.py

Reads the episodes record_demonstrations.py saved under config.RECORDINGS_ROOT and writes
the trained policy to config.POLICY_PATH, where run_policy.py looks for it.
"""

import logging

import config
from copy_cat.imitation_learning.discrete_action_imitation_learning import (
    DiscreteActionImitationLearning,
)
from copy_cat.policies.parameterized.discrete_action_cnn_policy import (
    DiscreteActionCNNPolicy,
)
from copy_cat.preprocessing.action_preprocessor import num_actions_from_names
from copy_cat.storage.trajectory_store import TrajectoryStore


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)

    store = TrajectoryStore(config.RECORDINGS_ROOT)
    episodes = store.list()

    if not episodes:
        raise SystemExit(
            f"No episodes in {str(config.RECORDINGS_ROOT)!r}; record one with "
            "record_demonstrations.py first."
        )

    trajectories = [store.load(episode) for episode in episodes]
    first = trajectories[0]

    observation_shape = first.get_step(0)[0].shape
    num_actions = num_actions_from_names(first.action_names)

    learner_policy = DiscreteActionCNNPolicy(
        observation_shape=observation_shape, num_actions=num_actions
    )
    algorithm = DiscreteActionImitationLearning(
        expert_trajectories=trajectories, learner_policy=learner_policy, optimizer=None
    )

    trained_policy = algorithm.train(num_epochs=config.NUM_EPOCHS)
    trained_policy.save(str(config.POLICY_PATH))

    print("Trained policy:", trained_policy)
    print("Saved to:", config.POLICY_PATH)


if __name__ == "__main__":
    main()
