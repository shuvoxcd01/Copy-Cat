import numpy as np
import pytest

from copy_cat.data.replay_buffer import SimpleReplayBuffer


def fill(buffer, count, start=0):
    """Store transitions whose fields all derive from their index, so order is checkable."""
    for index in range(start, start + count):
        buffer.store(
            state=index,
            action=index * 10,
            reward=float(index),
            next_state=index + 1,
            terminated=index % 2 == 0,
            truncated=False,
        )

    return buffer


@pytest.fixture
def buffer():
    return fill(SimpleReplayBuffer(), 5)


class TestStorage:
    def test_starts_empty(self):
        buffer = SimpleReplayBuffer()

        assert len(buffer) == 0
        assert buffer.size() == 0
        assert buffer.capacity is None

    def test_store_keeps_transitions_in_order(self, buffer):
        assert len(buffer) == 5
        assert buffer.buffer[0] == (0, 0, 0.0, 1, True, False)
        assert buffer.buffer[-1] == (4, 40, 4.0, 5, True, False)

    def test_capacity_evicts_the_oldest(self):
        buffer = fill(SimpleReplayBuffer(capacity=3), 5)

        assert len(buffer) == 3
        assert [transition[0] for transition in buffer.buffer] == [2, 3, 4]

    def test_extend_appends_another_buffer(self, buffer):
        other = fill(SimpleReplayBuffer(), 2, start=100)

        buffer.extend(other)

        assert len(buffer) == 7
        assert buffer.buffer[-1][0] == 101
        assert len(other) == 2

    def test_clear(self, buffer):
        buffer.clear()

        assert len(buffer) == 0

    def test_pop_removes_from_the_front(self, buffer):
        popped = buffer.pop(2)

        assert [transition[0] for transition in popped] == [0, 1]
        assert len(buffer) == 3
        assert buffer.pop()[0][0] == 2

    def test_pop_more_than_stored_is_rejected(self, buffer):
        with pytest.raises(ValueError, match="greater than buffer size"):
            buffer.pop(6)

        assert len(buffer) == 5


class TestSample:
    def test_returns_aligned_arrays(self, buffer):
        states, actions, rewards, next_states, terminated, truncated = buffer.sample(3)

        for array in (actions, rewards, terminated, truncated):
            assert isinstance(array, np.ndarray)
            assert array.shape == (3,)

        # Scalar states are reshaped into column vectors.
        assert states.shape == (3, 1)
        assert next_states.shape == (3, 1)

        # Every field of a sampled transition must come from the same transition.
        np.testing.assert_array_equal(actions, states[:, 0] * 10)
        np.testing.assert_array_equal(next_states[:, 0], states[:, 0] + 1)
        np.testing.assert_array_equal(rewards, states[:, 0].astype(float))

    def test_random_sample_has_no_duplicates(self, buffer):
        states = buffer.sample(5)[0][:, 0]

        assert sorted(states.tolist()) == [0, 1, 2, 3, 4]

    def test_sequential_sample_is_contiguous(self, buffer):
        for _ in range(20):
            states = buffer.sample(3, sequential=True)[0][:, 0]

            assert np.all(np.diff(states) == 1)
            assert 0 <= states[0] <= 2

    def test_vector_states_keep_their_shape(self):
        buffer = SimpleReplayBuffer()

        for index in range(3):
            buffer.store(np.full(4, index), 0, 0.0, np.full(4, index + 1), False, False)

        states, _, _, next_states, _, _ = buffer.sample(2)

        assert states.shape == (2, 4)
        assert next_states.shape == (2, 4)

    def test_default_batch_size_is_one(self, buffer):
        assert buffer.sample()[1].shape == (1,)

    def test_batch_larger_than_buffer_is_rejected(self, buffer):
        with pytest.raises(ValueError, match="greater than buffer size"):
            buffer.sample(6)
