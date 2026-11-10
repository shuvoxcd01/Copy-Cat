"""Observation preprocessing over episodes that went through a TrajectoryStore on disk."""

from functools import partial

import numpy as np
import pytest

from copy_cat.data.trajectory import Trajectory
from copy_cat.preprocessing.observation_preprocessor import ObservationPreprocessor
from copy_cat.storage.trajectory_store import TrajectoryStore

pytestmark = pytest.mark.integration

OP = ObservationPreprocessor


def make_frame(shape=(8, 8, 3), dtype=np.uint8, seed=0):
    """Build a frame that looks like a captured screen observation."""
    rng = np.random.default_rng(seed)

    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 255, size=shape, dtype=dtype)

    return rng.random(size=shape).astype(dtype)


class TestStorageRoundTrip:
    def test_a_preprocessed_episode_still_saves_and_loads(self, tmp_path):
        pipeline = OP.compose(OP.to_grayscale, partial(OP.downsample, factor=2))
        trajectory = Trajectory()

        for step in range(3):
            trajectory.record_step(
                state=pipeline(make_frame(shape=(16, 16, 3), seed=step)),
                action=step,
                reward=None,
                timestamp=round(step * 0.1, 3),
            )

        store = TrajectoryStore(tmp_path)
        name = store.save(trajectory, fps=10)
        loaded = store.load(name)

        assert store.info(name)["obs_shape"] == [8, 8]
        assert store.info(name)["obs_dtype"] == "uint8"

        for step in range(3):
            assert np.array_equal(loaded.get_state(step), trajectory.get_state(step))

    def test_a_pipeline_runs_over_read_only_memmapped_frames(self, tmp_path):
        trajectory = Trajectory()

        for step in range(3):
            trajectory.record_step(
                state=make_frame(shape=(16, 16, 3), seed=step),
                action=step,
                reward=None,
            )

        store = TrajectoryStore(tmp_path)
        name = store.save(trajectory)
        loaded = store.load(name, mmap=True)

        stored = loaded.get_state(0)

        assert not stored.flags.writeable

        pipeline = OP.compose(OP.to_grayscale, partial(OP.downsample, factor=2), OP.normalize)

        assert pipeline(stored).shape == (8, 8)
