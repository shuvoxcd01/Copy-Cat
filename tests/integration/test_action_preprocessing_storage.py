"""Action preprocessing over episodes that went through a TrajectoryStore on disk."""

from functools import partial

import numpy as np
import pytest

from copy_cat.data.trajectory import Trajectory
from copy_cat.preprocessing.action_preprocessor import (
    ENCODING_DTYPE,
    ActionPreprocessor,
    num_actions_from_names,
)
from copy_cat.storage.trajectory_store import TrajectoryStore

pytestmark = pytest.mark.integration

AP = ActionPreprocessor


def make_trajectory(actions, shape=(4, 4), action_names=("space",), seed=0):
    """Build a trajectory carrying the given actions, for a store round trip."""
    rng = np.random.default_rng(seed)
    trajectory = Trajectory()

    for step, action in enumerate(actions):
        frame = rng.integers(0, 255, size=shape, dtype=np.uint8)
        trajectory.record_step(state=frame, action=int(action), timestamp=step / 10.0)

    trajectory.action_names = action_names

    return trajectory


class TestStorageRoundTrip:
    def test_actions_come_back_off_a_store_ready_to_encode(self, tmp_path):
        """Actions loaded from a store are plain ints, which must encode without a cast."""
        store = TrajectoryStore(tmp_path)
        name = store.save(make_trajectory([0, 1, 0, 1, 1]), fps=10.0)
        loaded = store.load(name)

        actions = [step[1] for step in loaded.state_actions]

        assert isinstance(actions[0], int)

        encoded = AP.to_onehot(actions, num_actions_from_names(loaded.action_names))

        assert encoded.shape == (5, 2)
        assert encoded.dtype == ENCODING_DTYPE
        assert np.array_equal(encoded.sum(axis=-1), np.ones(5))

    def test_an_episode_with_no_key_press_still_encodes_both_classes(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        name = store.save(make_trajectory([0, 0, 0, 0]), fps=10.0)
        loaded = store.load(name)

        actions = [step[1] for step in loaded.state_actions]
        num_classes = num_actions_from_names(loaded.action_names)

        assert num_classes == 2
        assert max(actions) + 1 == 1  # what inferring num_classes would have given
        assert AP.to_onehot(actions, num_classes).shape == (4, 2)

    def test_action_names_survive_the_round_trip_for_to_names(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        name = store.save(make_trajectory([0, 1]), fps=10.0)
        loaded = store.load(name)

        actions = [step[1] for step in loaded.state_actions]

        assert AP.to_names(actions, loaded.action_names).tolist() == ["noop", "space"]

    def test_a_pipeline_runs_over_a_read_only_episode_from_a_store(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        name = store.save(make_trajectory([0, 1, 0, 1, 0, 0]), fps=10.0)
        loaded = store.load(name)

        actions = np.asarray([step[1] for step in loaded.state_actions])
        actions.flags.writeable = False
        original = actions.copy()

        pipeline = AP.compose(
            partial(AP.hold, max_hold=2),
            partial(AP.action_history, length=4),
            partial(AP.to_onehot, num_classes=2),
            AP.flatten_history,
        )
        result = pipeline(actions)

        assert result.shape == (6, 8)
        assert np.array_equal(actions, original)
