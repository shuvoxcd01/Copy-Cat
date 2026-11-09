from functools import partial

import numpy as np
import pytest

from copy_cat.storage.trajectory_store import TrajectoryStore
from copy_cat.utils.action_preprocessor import (
    ENCODING_DTYPE,
    INDEX_DTYPE,
    NOOP_ACTION,
    NOOP_NAME,
    ActionPreprocessor,
    mapping_from_names,
    num_actions_from_names,
)
from copy_cat.utils.trajectory import Trajectory

AP = ActionPreprocessor


def make_actions(length=16, num_classes=3, noop_rate=0.7, seed=0):
    """Build an episode of actions that looks like a recording: mostly noop."""
    rng = np.random.default_rng(seed)
    pressed = rng.integers(1, num_classes, size=length)
    idle = rng.random(length) < noop_rate

    return np.where(idle, 0, pressed).astype(np.int64)


def make_trajectory(actions, shape=(4, 4), action_names=("space",), seed=0):
    """Build a trajectory carrying the given actions, for a store round trip."""
    rng = np.random.default_rng(seed)
    trajectory = Trajectory()

    for step, action in enumerate(actions):
        frame = rng.integers(0, 255, size=shape, dtype=np.uint8)
        trajectory.record_step(state=frame, action=int(action), timestamp=step / 10.0)

    trajectory.action_names = action_names

    return trajectory


class TestCoercion:
    def test_accepts_a_bare_int_from_a_capture(self):
        encoded = AP.to_onehot(1, num_classes=3)

        assert encoded.shape == (3,)
        assert np.array_equal(encoded, [0.0, 1.0, 0.0])

    def test_accepts_a_list_of_ints_from_a_loaded_trajectory(self):
        encoded = AP.to_onehot([0, 2, 1], num_classes=3)

        assert encoded.shape == (3, 3)
        assert np.array_equal(np.argmax(encoded, axis=-1), [0, 2, 1])

    def test_normalises_every_index_array_to_int64(self):
        actions = np.asarray([0, 1, 2], dtype=np.int8)

        assert AP.hold(actions).dtype == INDEX_DTYPE
        assert AP.add_batch_dim(actions).dtype == INDEX_DTYPE

    def test_rejects_a_bool_array_that_would_become_action_one(self):
        with pytest.raises(TypeError, match="boolean"):
            AP.hold(np.asarray([True, False]))

    def test_rejects_a_float_array_that_is_probably_an_encoding(self):
        with pytest.raises(TypeError, match="to_index"):
            AP.hold(np.asarray([0.0, 1.0]))

    def test_rejects_a_string_and_an_object_array(self):
        with pytest.raises(TypeError, match="integer action indices"):
            AP.hold(np.asarray(["space", "noop"]))

        with pytest.raises(TypeError, match="integer action indices"):
            AP.hold(np.asarray([{"a": 1}], dtype=object))

    def test_rejects_none(self):
        with pytest.raises(TypeError, match="got None"):
            AP.hold(None)

    def test_rejects_a_negative_index(self):
        with pytest.raises(ValueError, match="must not be negative, got -1"):
            AP.hold(np.asarray([0, -1]))

    def test_rejects_a_rank_that_is_neither_an_action_nor_an_episode(self):
        with pytest.raises(ValueError, match="history block"):
            AP.to_onehot(np.zeros((2, 2, 2), dtype=np.int64), num_classes=2)

    def test_a_temporal_preprocessor_wants_an_episode_not_one_action(self):
        with pytest.raises(ValueError, match="episode"):
            AP.hold(1)


class TestEncoding:
    def test_onehot_puts_a_single_one_in_the_action_column(self):
        encoded = AP.to_onehot([0, 1, 2, 1], num_classes=3)

        assert encoded.shape == (4, 3)
        assert np.array_equal(encoded.sum(axis=-1), np.ones(4))
        assert encoded[2, 2] == 1.0

    def test_onehot_of_one_action_has_no_leading_axis(self):
        assert AP.to_onehot(1, num_classes=4).shape == (4,)

    def test_onehot_of_a_history_block_appends_the_class_axis(self):
        history = AP.action_history(make_actions(), length=3)

        assert AP.to_onehot(history, num_classes=3).shape == (16, 3, 3)

    def test_onehot_is_float32_by_default_and_honours_a_dtype(self):
        assert AP.to_onehot([0, 1], num_classes=2).dtype == ENCODING_DTYPE
        assert AP.to_onehot([0, 1], num_classes=2, dtype=np.uint8).dtype == np.uint8

    def test_onehot_keeps_a_column_for_a_key_that_was_never_pressed(self):
        """A recording where the key is never touched must still encode both classes."""
        encoded = AP.to_onehot([0, 0, 0], num_classes=2)

        assert encoded.shape == (3, 2)
        assert np.array_equal(encoded[:, 1], np.zeros(3))

    def test_onehot_rejects_an_index_at_or_above_num_classes(self):
        with pytest.raises(ValueError, match="num_classes 2, got 5"):
            AP.to_onehot([0, 5], num_classes=2)

    def test_onehot_rejects_a_class_count_below_one(self):
        with pytest.raises(ValueError, match="num_classes must be at least 1"):
            AP.to_onehot([0], num_classes=0)

    def test_to_index_inverts_to_onehot(self):
        actions = make_actions()

        assert np.array_equal(AP.to_index(AP.to_onehot(actions, 3)), actions)

    def test_to_index_of_one_encoded_action_gives_a_zero_dimensional_array(self):
        index = AP.to_index(AP.to_onehot(2, num_classes=4))

        assert index.shape == ()
        assert int(index) == 2

    def test_to_index_picks_the_largest_probability(self):
        logits = np.asarray([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])

        assert np.array_equal(AP.to_index(logits), [1, 0])

    def test_to_index_breaks_a_tie_towards_the_noop(self):
        assert int(AP.to_index(np.asarray([0.5, 0.5]))) == NOOP_ACTION

    def test_to_index_rejects_a_scalar_with_no_class_axis(self):
        with pytest.raises(ValueError, match="encoded action"):
            AP.to_index(1)

    def test_label_smooth_rows_sum_to_one(self):
        smoothed = AP.label_smooth(make_actions(), num_classes=3, epsilon=0.2)

        assert np.allclose(smoothed.sum(axis=-1), 1.0)
        assert smoothed.dtype == ENCODING_DTYPE

    def test_label_smooth_spreads_epsilon_over_every_class(self):
        smoothed = AP.label_smooth(1, num_classes=4, epsilon=0.4)

        assert np.allclose(smoothed, [0.1, 0.7, 0.1, 0.1])

    def test_label_smooth_with_no_epsilon_is_a_onehot(self):
        actions = make_actions()

        assert np.array_equal(
            AP.label_smooth(actions, 3, epsilon=0.0), AP.to_onehot(actions, 3)
        )

    def test_label_smooth_rejects_an_epsilon_outside_zero_to_one(self):
        with pytest.raises(ValueError, match="epsilon must be between 0 and 1"):
            AP.label_smooth([0, 1], num_classes=2, epsilon=1.5)

    def test_label_smooth_rejects_an_integer_dtype(self):
        with pytest.raises(TypeError, match="float dtype"):
            AP.label_smooth([0, 1], num_classes=2, dtype=np.int64)


class TestVocabulary:
    def test_remap_relabels_through_the_mapping(self):
        remapped = AP.remap([0, 1, 2], mapping={0: 0, 1: 2, 2: 1})

        assert np.array_equal(remapped, [0, 2, 1])
        assert remapped.dtype == INDEX_DTYPE

    def test_remap_keeps_the_shape_of_one_action(self):
        remapped = AP.remap(1, mapping={0: 0, 1: 5})

        assert remapped.shape == ()
        assert int(remapped) == 5

    def test_remap_rejects_an_index_the_mapping_does_not_cover(self):
        with pytest.raises(ValueError, match="no entry for actions \\[2\\]"):
            AP.remap([0, 1, 2], mapping={0: 0, 1: 1})

    def test_remap_sends_uncovered_indices_to_the_default(self):
        remapped = AP.remap([0, 1, 2], mapping={0: 0, 1: 1}, default=NOOP_ACTION)

        assert np.array_equal(remapped, [0, 1, 0])

    def test_remap_rejects_something_that_is_not_a_mapping(self):
        with pytest.raises(TypeError, match="mapping of index to index"):
            AP.remap([0, 1], mapping=[(0, 1)])

    def test_to_names_labels_the_noop_the_way_playback_does(self):
        labelled = AP.to_names([0, 1], action_names=("space",))

        assert labelled.tolist() == [NOOP_NAME, "space"]

    def test_to_names_uses_the_key_list_that_produced_the_indices(self):
        labelled = AP.to_names([2, 0, 1], action_names=("w", "a"))

        assert labelled.tolist() == ["a", "noop", "w"]

    def test_to_names_rejects_a_missing_key_list(self):
        with pytest.raises(TypeError, match="action_names"):
            AP.to_names([0, 1], action_names=None)

    def test_to_names_rejects_an_index_the_key_list_cannot_label(self):
        with pytest.raises(ValueError, match="must index the key list"):
            AP.to_names([0, 3], action_names=("space",))

    def test_to_names_rejects_a_noop_name_that_collides_with_a_key(self):
        with pytest.raises(ValueError, match="also a key"):
            AP.to_names([0], action_names=("noop",))

    def test_from_names_inverts_to_names(self):
        actions = make_actions(num_classes=3)
        names = ("w", "a")

        assert np.array_equal(AP.from_names(AP.to_names(actions, names), names), actions)

    def test_from_names_rejects_a_key_that_was_never_recorded(self):
        with pytest.raises(ValueError, match="does not know"):
            AP.from_names(["space", "escape"], action_names=("space",))

    def test_num_actions_from_names_counts_the_noop_that_meta_leaves_out(self):
        assert num_actions_from_names(("space",)) == 2
        assert num_actions_from_names(("w", "a", "s", "d")) == 5

    def test_num_actions_from_names_rejects_a_missing_key_list(self):
        with pytest.raises(TypeError, match="trajectory.action_names"):
            num_actions_from_names(None)

    def test_num_actions_from_names_rejects_a_repeated_key(self):
        with pytest.raises(ValueError, match="must not repeat a key"):
            num_actions_from_names(("w", "w"))

    def test_mapping_from_names_lines_two_key_orders_up(self):
        mapping = mapping_from_names(("w", "a"), ("a", "w"))

        assert mapping == {0: 0, 1: 2, 2: 1}
        assert np.array_equal(AP.remap([1, 2], mapping), [2, 1])

    def test_mapping_from_names_sends_a_dropped_key_to_the_noop(self):
        mapping = mapping_from_names(("w", "a"), ("w",))

        assert mapping == {0: 0, 1: 1, 2: NOOP_ACTION}


class TestTemporal:
    def test_hold_carries_the_last_action_across_a_noop_gap(self):
        assert np.array_equal(AP.hold([0, 1, 0, 0, 2, 0]), [0, 1, 1, 1, 2, 2])

    def test_hold_leaves_the_noops_before_the_first_action_alone(self):
        assert np.array_equal(AP.hold([0, 0, 1, 0]), [0, 0, 1, 1])

    def test_hold_stops_carrying_after_max_hold_steps(self):
        assert np.array_equal(AP.hold([1, 0, 0, 0], max_hold=2), [1, 1, 1, 0])

    def test_hold_honours_a_different_noop(self):
        assert np.array_equal(AP.hold([9, 1, 9], noop=9), [9, 1, 1])

    def test_hold_rejects_a_max_hold_below_one(self):
        with pytest.raises(ValueError, match="max_hold must be at least 1"):
            AP.hold([0, 1], max_hold=0)

    def test_shift_lags_the_episode_and_fills_the_start(self):
        assert np.array_equal(AP.shift([1, 2, 3], steps=1), [0, 1, 2])

    def test_shift_with_a_negative_step_looks_ahead(self):
        assert np.array_equal(AP.shift([1, 2, 3], steps=-1), [2, 3, 0])

    def test_shift_by_more_than_the_episode_is_all_fill(self):
        assert np.array_equal(AP.shift([1, 2], steps=5, fill=7), [7, 7])

    def test_shift_by_zero_returns_the_episode(self):
        actions = make_actions()

        assert np.array_equal(AP.shift(actions, steps=0), actions)

    def test_shift_rejects_a_non_integer_step(self):
        with pytest.raises(TypeError, match="steps must be an integer"):
            AP.shift([0, 1], steps=1.5)

    def test_action_history_excludes_the_action_it_would_be_predicting(self):
        """In behaviour cloning the current action is the label, so it must not leak."""
        history = AP.action_history([1, 2, 3], length=2)

        assert np.array_equal(history, [[0, 0], [0, 1], [1, 2]])

    def test_action_history_includes_the_current_action_when_asked(self):
        history = AP.action_history([1, 2, 3], length=2, include_current=True)

        assert np.array_equal(history, [[0, 1], [1, 2], [2, 3]])

    def test_action_history_pads_the_first_steps_with_the_fill(self):
        history = AP.action_history([1, 2], length=3, fill=9)

        assert np.array_equal(history[0], [9, 9, 9])

    def test_action_history_rows_run_oldest_first(self):
        history = AP.action_history([1, 2, 3, 4], length=3, include_current=True)

        assert np.array_equal(history[-1], [2, 3, 4])

    def test_action_history_rejects_a_length_below_one(self):
        with pytest.raises(ValueError, match="length must be at least 1"):
            AP.action_history([0, 1], length=0)


class TestLayout:
    def test_add_batch_dim_turns_one_action_into_a_one_step_episode(self):
        assert AP.add_batch_dim(1).shape == (1,)

    def test_add_batch_dim_leaves_an_episode_alone(self):
        actions = make_actions()

        assert np.array_equal(AP.add_batch_dim(actions), actions)

    def test_to_column_gives_one_row_per_step(self):
        column = AP.to_column([0, 1, 2])

        assert column.shape == (3, 1)
        assert np.array_equal(column[:, 0], [0, 1, 2])

    def test_to_column_rejects_a_history_block(self):
        with pytest.raises(ValueError, match="one row per step"):
            AP.to_column(np.zeros((3, 2), dtype=np.int64))

    def test_flatten_history_keeps_the_oldest_action_first(self):
        history = AP.action_history([1, 2, 3], length=2, include_current=True)
        flat = AP.flatten_history(AP.to_onehot(history, num_classes=4))

        assert flat.shape == (3, 8)
        # The last row is [2, 3], so the first four columns encode 2 and the rest encode 3.
        assert np.array_equal(flat[-1], [0, 0, 1, 0, 0, 0, 0, 1])

    def test_flatten_history_rejects_anything_but_an_encoded_history(self):
        with pytest.raises(ValueError, match="encoded history"):
            AP.flatten_history(AP.to_onehot([0, 1], num_classes=2))


class TestCompose:
    def test_applies_preprocessors_in_order(self):
        pipeline = AP.compose(
            AP.hold,
            partial(AP.action_history, length=2),
            partial(AP.to_onehot, num_classes=3),
            AP.flatten_history,
        )
        result = pipeline(make_actions())

        assert result.shape == (16, 6)
        assert result.dtype == ENCODING_DTYPE

    def test_an_empty_pipeline_is_the_identity(self):
        actions = make_actions()

        assert AP.compose()(actions) is actions

    def test_rejects_a_non_callable(self):
        with pytest.raises(ValueError, match="compose takes callables"):
            AP.compose(AP.hold, "hold")


HISTORY = partial(AP.action_history, length=3)

EPISODE_PREPROCESSORS = [
    AP.hold,
    partial(AP.hold, max_hold=2),
    partial(AP.shift, steps=1),
    partial(AP.shift, steps=-2),
    partial(AP.shift, steps=0),
    HISTORY,
    partial(AP.remap, mapping={0: 0, 1: 2, 2: 1}),
    partial(AP.to_onehot, num_classes=3),
    partial(AP.label_smooth, num_classes=3),
    partial(AP.to_names, action_names=("space", "up")),
    AP.add_batch_dim,
    AP.to_column,
]

# to_names returns key names rather than indices, so only the rest carry the int64 rule.
INDEX_PREPROCESSORS = [
    AP.hold,
    partial(AP.hold, max_hold=2),
    partial(AP.shift, steps=1),
    partial(AP.shift, steps=0),
    HISTORY,
    partial(AP.remap, mapping={0: 0, 1: 2, 2: 1}),
    AP.add_batch_dim,
    AP.to_column,
]

# hold, shift and action_history read neighbouring steps, so only these can match an
# episode processed one action at a time.
ELEMENTWISE_PREPROCESSORS = [
    partial(AP.remap, mapping={0: 0, 1: 2, 2: 1}),
    partial(AP.to_onehot, num_classes=3),
    partial(AP.label_smooth, num_classes=3),
    partial(AP.to_names, action_names=("space", "up")),
]


@pytest.mark.parametrize("preprocessor", EPISODE_PREPROCESSORS)
def test_no_preprocessor_writes_to_its_input(preprocessor):
    """Actions live inside the caller's trajectory, so nothing may write in place."""
    actions = make_actions()
    actions.flags.writeable = False
    original = actions.copy()

    result = preprocessor(actions)

    assert np.array_equal(actions, original)

    # A pass-through may hand the input straight back; anything else must be a new array.
    if result.shape == actions.shape and result.dtype == actions.dtype:
        assert result is actions or not np.shares_memory(result, actions)


@pytest.mark.parametrize("preprocessor", INDEX_PREPROCESSORS)
def test_every_index_preprocessor_returns_int64(preprocessor):
    assert preprocessor(make_actions().astype(np.int16)).dtype == INDEX_DTYPE


@pytest.mark.parametrize("preprocessor", ELEMENTWISE_PREPROCESSORS)
def test_an_episode_matches_processing_one_action_at_a_time(preprocessor):
    actions = make_actions()

    together = preprocessor(actions)
    separately = np.stack([preprocessor(int(action)) for action in actions])

    assert np.array_equal(together, separately)


@pytest.mark.parametrize("preprocessor", EPISODE_PREPROCESSORS)
def test_every_preprocessor_rejects_a_float_episode(preprocessor):
    with pytest.raises(TypeError):
        preprocessor(make_actions().astype(np.float32))


@pytest.mark.parametrize("preprocessor", EPISODE_PREPROCESSORS)
def test_every_preprocessor_rejects_a_bool_episode(preprocessor):
    with pytest.raises(TypeError, match="boolean"):
        preprocessor(make_actions().astype(bool))


class TestStorageRoundTrip:
    def test_actions_come_back_off_a_store_ready_to_encode(self, tmp_path):
        """replay.py pulls a plain list of ints, which must encode without a manual cast."""
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
