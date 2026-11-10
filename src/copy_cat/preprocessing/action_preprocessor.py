from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

# KeyboardActionCapture reserves 0 for "no tracked key was active" and maps keys[i] to
# i + 1, so the noop is always the first class and action_names never carries it.
NOOP_ACTION = 0

# The label TrajectoryStore draws over a noop frame during playback, kept in step here.
NOOP_NAME = "noop"

# Indices stay int64 because that is what a cross-entropy target must be, and because
# np.asarray on a list of Python ints is only 32 bits wide on some platforms.
INDEX_DTYPE = np.int64

# Encodings default to float32, matching ObservationPreprocessor.normalize, so the
# observation and the action reach a network as one dtype.
ENCODING_DTYPE = np.float32


class ActionPreprocessor:
    """A library of static, cascadable transforms for recorded actions.

    This is the counterpart to ObservationPreprocessor, for the other half of a
    demonstration. Actions come from KeyboardActionCapture as discrete indices: 0 is the
    noop, keys[i] is i + 1, and only one key is reported per poll. That is the whole
    scope here - a single-key discrete space. A future multi-key space slots into the
    (N, K) shape action_history already emits, and a continuous space would be a
    different module.

    Actions carry two forms, and every method says which one it takes, so a rank-1 array
    is never guessed to be one thing or the other:

        index form                          encoded form
        ()          one action              (C,)        one encoded action
        (N,)        an episode              (N, C)      an encoded episode
        (N, K)      a history block         (N, K, C)   an encoded history

    The class axis is always last and the time axis is always first. The methods that
    read neighbouring steps - hold, shift, action_history - are tighter still and take
    an episode only, because one action has no history and a history block has no single
    time axis.

    Unlike ObservationPreprocessor, which refuses anything that is not already an array,
    this coerces: capture_action() returns a bare int and a trajectory loaded from a
    store hands back a list of them, so np.asarray belongs here rather than at every call
    site. Booleans, floats, strings and None are still refused, because each of those is
    a mistake rather than an action.

    num_classes is always required and never inferred from the data. A recording where a
    key was never pressed would make max(actions) + 1 under-count the action space, and a
    policy head of the wrong width fails far from its cause. Take it from the key list
    instead, with num_actions_from_names(trajectory.action_names) or capture.num_actions.

    Nothing here mutates its argument. Coercion can hand back the very array it was given,
    so an in-place write would reach straight into the caller's recorded actions.

    Preprocessors that take arguments are bound with functools.partial and assembled with
    compose:

        from functools import partial
        from copy_cat.preprocessing.action_preprocessor import ActionPreprocessor as AP

        pipeline = AP.compose(
            partial(AP.hold, max_hold=3),
            partial(AP.action_history, length=4),
            partial(AP.to_onehot, num_classes=2),
            AP.flatten_history,
        )
        actions = pipeline(episode)        # (77,) int64 -> (77, 8) float32

    Everything is pure numpy and works on a base install. to_names is the one exception to
    the actions-in rule, since it returns key names rather than indices and so ends a
    pipeline rather than continuing one.
    """

    # ---------------------------------------------------------------- encoding

    @staticmethod
    def to_onehot(
        actions: Any,
        num_classes: int,
        dtype: Any = ENCODING_DTYPE,
    ) -> np.ndarray:
        """Turn action indices into one column per action, for a policy head.

        A class axis is appended to whatever it is given, so one action becomes (C,), an
        episode becomes (N, C) and a history block becomes (N, K, C). num_classes is
        deliberately required: derive it from the key list with num_actions_from_names,
        never from the actions themselves.
        """
        indices = _as_indices(actions)
        classes = _as_num_classes(num_classes)
        resolved = np.dtype(dtype)

        _check_range(indices, classes)

        encoded = np.zeros((*indices.shape, classes), dtype=resolved)
        np.put_along_axis(encoded, indices[..., np.newaxis], 1, axis=-1)

        return encoded

    @staticmethod
    def to_index(encoded: Any) -> np.ndarray:
        """Take the largest value along the class axis, inverting to_onehot.

        This also reads a greedy action out of a policy's logits or probabilities. The
        class axis is consumed, so (C,) becomes a 0-d array, (N, C) becomes (N,) and
        (N, K, C) becomes (N, K). Ties go to the lowest index, which biases a dead-flat
        output towards the noop - the safe choice for a demonstration policy.
        """
        values = _as_encoded(encoded)

        return np.asarray(np.argmax(values, axis=-1)).astype(INDEX_DTYPE)

    @staticmethod
    def label_smooth(
        actions: Any,
        num_classes: int,
        epsilon: float = 0.1,
        dtype: Any = ENCODING_DTYPE,
    ) -> np.ndarray:
        """Encode actions as a softened one-hot, for a demonstrator who was not certain.

        Cross-entropy against a hard one-hot pushes a behaviour-cloned policy towards
        infinite confidence on a human who was hesitating. This spreads epsilon evenly
        over every class - (1 - epsilon) * onehot + epsilon / num_classes - so each row
        still sums to one. Shapes match to_onehot exactly.
        """
        resolved = np.dtype(dtype)

        if not np.issubdtype(resolved, np.floating):
            raise TypeError(f"label_smooth needs a float dtype, got {resolved}.")

        spread = float(epsilon)

        if not 0.0 <= spread <= 1.0:
            raise ValueError(f"epsilon must be between 0 and 1, got {spread}.")

        classes = _as_num_classes(num_classes)
        encoded = ActionPreprocessor.to_onehot(actions, classes, dtype=resolved)

        return encoded * resolved.type(1.0 - spread) + resolved.type(spread / classes)

    # -------------------------------------------------------------- vocabulary

    @staticmethod
    def remap(
        actions: Any,
        mapping: Mapping,
        default: Optional[int] = None,
    ) -> np.ndarray:
        """Relabel action indices through a mapping, keeping the shape.

        This merges several keys into one action, or lines two recordings up on a single
        vocabulary. Without a default every index present must be covered, so a key the
        mapping forgot is an error rather than a silent relabel; build the mapping from
        two key lists with mapping_from_names.
        """
        indices = _as_indices(actions)
        table_map = _as_mapping(mapping)

        if default is None:
            missing = sorted(set(np.unique(indices).tolist()) - set(table_map))

            if missing:
                raise ValueError(
                    f"remap has no entry for actions {missing}; pass a default."
                )

            fill = NOOP_ACTION
        else:
            fill = _as_action(default, "default")

        size = int(max(int(indices.max(initial=0)), max(table_map), fill)) + 1

        return np.asarray(_lookup_table(table_map, fill, size)[indices])

    @staticmethod
    def to_names(
        actions: Any,
        action_names: Sequence[str],
        noop: str = NOOP_NAME,
    ) -> np.ndarray:
        """Label action indices with the key names that produced them.

        This is for logging, plots and eyeballing a recording: it uses the same 0 -> noop,
        i -> action_names[i - 1] convention TrajectoryStore draws over a played-back
        frame, so a printed episode reads the same as the overlay. The result holds
        strings rather than indices, so it ends a pipeline rather than continuing one.
        """
        indices = _as_indices(actions)
        labels = _labels(action_names, noop)

        if indices.size and int(indices.max()) >= len(labels):
            raise ValueError(
                f"actions must index the key list {list(labels[1:])}, got "
                f"{int(indices.max())}."
            )

        return np.asarray(np.asarray(labels)[indices])

    @staticmethod
    def from_names(
        names: Any,
        action_names: Sequence[str],
        noop: str = NOOP_NAME,
    ) -> np.ndarray:
        """Turn key names back into action indices, inverting to_names.

        This is how an expected action sequence gets written out by hand, in the names a
        recording was made with rather than in the integers they happen to map to.
        """
        labels = _labels(action_names, noop)
        lookup = {label: index for index, label in enumerate(labels)}
        array = np.asarray(names)

        if array.dtype.kind not in "US":
            raise TypeError(f"names must hold key names, got dtype {array.dtype}.")

        flat = [str(entry) for entry in array.reshape(-1)]
        unknown = sorted(set(flat) - set(lookup))

        if unknown:
            raise ValueError(
                f"from_names does not know {unknown}; the vocabulary is {list(labels)}."
            )

        indices = np.asarray([lookup[entry] for entry in flat], dtype=INDEX_DTYPE)

        return indices.reshape(array.shape)

    # ---------------------------------------------------------------- temporal

    @staticmethod
    def hold(
        actions: Any,
        noop: int = NOOP_ACTION,
        max_hold: Optional[int] = None,
    ) -> np.ndarray:
        """Carry the last action forward across noop steps, so a held key stays held.

        A human holding a key down is intent, not idleness, but a poll that lands between
        two key events records a noop anyway. Leading noops, before anything was pressed,
        stay noop - there is nothing to carry. max_hold caps how many consecutive noop
        steps get filled, because past a second or so a held key and a genuinely idle
        stretch are indistinguishable and filling forever manufactures labels.

        This is not the ALE "sticky actions" of injected stochasticity, which is why it
        is not called that.
        """
        indices = _as_episode(actions)
        idle = _as_action(noop, "noop")

        if max_hold is not None:
            max_hold = int(max_hold)

            if max_hold < 1:
                raise ValueError(f"max_hold must be at least 1, got {max_hold}.")

        if indices.size == 0:
            return indices

        steps = np.arange(len(indices))

        # The position of the most recent real action at or before each step, -1 before
        # the first one, which is what makes the leading noops stay noop.
        last = np.maximum.accumulate(np.where(indices != idle, steps, -1))
        held = np.where(last >= 0, indices[last], INDEX_DTYPE(idle))

        if max_hold is not None:
            held = np.where(steps - last > max_hold, INDEX_DTYPE(idle), held)

        return held.astype(INDEX_DTYPE)

    @staticmethod
    def shift(actions: Any, steps: int = 1, fill: int = NOOP_ACTION) -> np.ndarray:
        """Move an episode of actions in time, keeping its length.

        A positive steps lags the episode, so step i receives the action from i - steps
        and the start is filled - that is both the correction for a screen capture that
        lands after the key that caused it, and the way to build "the previous action". A
        negative steps looks ahead instead. Shifting by more than the episode is legal and
        gives an episode of nothing but fill.
        """
        indices = _as_episode(actions)

        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
            raise TypeError(f"steps must be an integer, got {type(steps).__name__}.")

        steps = int(steps)
        pad = _as_action(fill, "fill")

        if steps == 0:
            return indices

        shifted = np.full(indices.shape, pad, dtype=INDEX_DTYPE)

        if abs(steps) >= len(indices):
            return shifted

        if steps > 0:
            shifted[steps:] = indices[:-steps]
        else:
            shifted[:steps] = indices[-steps:]

        return shifted

    @staticmethod
    def action_history(
        actions: Any,
        length: int = 4,
        fill: int = NOOP_ACTION,
        include_current: bool = False,
    ) -> np.ndarray:
        """Give every step the actions that came before it, oldest first.

        This is the framestack analogue for actions: an episode of (N,) becomes
        (N, length), so a policy can be conditioned on what the demonstrator was just
        doing. The first steps have no history and are padded with fill.

        include_current is False by default and that default is load-bearing. In behaviour
        cloning the action at step i is the label, so a window that includes it hands the
        network the answer. Set it to True only when the actions are inputs rather than
        targets; that window matches ObservationPreprocessor.framestack exactly.

        The (N, K) result is where this band ends - hold and shift take an episode only -
        but it feeds to_onehot, giving (N, K, C) for flatten_history. Like
        to_channels_first, it belongs at the end of a pipeline.
        """
        indices = _as_episode(actions)
        length = int(length)

        if length < 1:
            raise ValueError(f"length must be at least 1, got {length}.")

        pad = _as_action(fill, "fill")
        count = len(indices)

        if count == 0:
            return np.zeros((0, length), dtype=INDEX_DTYPE)

        end = 1 if include_current else 0
        offsets = np.arange(-length + end, end)
        positions = np.arange(count)[:, np.newaxis] + offsets
        taken = indices[np.clip(positions, 0, count - 1)]

        return np.where(positions >= 0, taken, INDEX_DTYPE(pad)).astype(INDEX_DTYPE)

    # ------------------------------------------------------------------ layout

    @staticmethod
    def add_batch_dim(actions: Any) -> np.ndarray:
        """Turn one action into a one-step episode, so a capture takes the episode path.

        An episode is returned unchanged, so this can be used to guarantee a rank. There
        is no encoded-form equivalent on purpose: batching before encoding gives the same
        (1, C) without anyone having to guess whether a rank-1 array is (N,) or (C,).
        Batch first, then encode.
        """
        indices = _as_indices(actions)

        if indices.ndim == 0:
            return indices[np.newaxis]

        return indices

    @staticmethod
    def to_column(actions: Any) -> np.ndarray:
        """Stand an episode of actions up as one column, for a head or a concatenation."""
        indices = _as_indices(actions)

        if indices.ndim == 0:
            return indices.reshape(1, 1)

        if indices.ndim == 1:
            return indices.reshape(-1, 1)

        raise ValueError(
            f"to_column takes one action or an episode, got shape {indices.shape}; a "
            "history block already has one row per step."
        )

    @staticmethod
    def flatten_history(encoded: Any) -> np.ndarray:
        """Collapse an encoded action history to one row per step, for an MLP.

        This takes (N, K, C) and returns (N, K * C), oldest action first. It is rank-3
        only rather than a general flatten, because a rank-2 array is genuinely ambiguous
        between an encoded episode, which should pass through, and one encoded history,
        which should flatten - and this class does not guess.
        """
        values = _as_encoded(encoded)

        if values.ndim != 3:
            raise ValueError(
                f"flatten_history takes an encoded history (N, K, C), got shape "
                f"{values.shape}."
            )

        count, history, classes = values.shape

        return values.reshape(count, history * classes)

    # ---------------------------------------------------------------- plumbing

    @staticmethod
    def compose(*preprocessors: Callable[[Any], np.ndarray]) -> Callable:
        """Chain preprocessors into one callable, applied left to right.

        Bind arguments with functools.partial, so the pipeline is built once and reused
        for every episode of a dataset.
        """
        for index, preprocessor in enumerate(preprocessors):
            if not callable(preprocessor):
                raise ValueError(
                    f"compose takes callables; argument {index} is a "
                    f"{type(preprocessor).__name__}."
                )

        def pipeline(actions: Any) -> np.ndarray:
            """Apply every composed preprocessor to an episode of actions, in order."""
            for preprocessor in preprocessors:
                actions = preprocessor(actions)

            return actions

        return pipeline


def num_actions_from_names(action_names: Optional[Sequence[str]]) -> int:
    """Return the size of the action space a key list produces, counting the noop.

    meta.json stores action_names without the noop, because KeyboardActionCapture reserves
    0 for it, so this owns that off-by-one in one place. It is the right argument for
    every num_classes in this module.
    """
    return len(_as_names(action_names)) + 1


def mapping_from_names(
    source_names: Sequence[str],
    target_names: Sequence[str],
    default: int = NOOP_ACTION,
) -> Dict[int, int]:
    """Build a remap mapping that carries one key list's indices onto another's.

    Two recordings made with ["w", "a", "s", "d"] and ["a", "d", "w"] use incompatible
    integers for the same keys. This lines them up, keeping the noop offset in one place;
    a source key the target does not have goes to default.
    """
    source = _as_names(source_names, "source_names")
    target = _as_names(target_names, "target_names")
    fill = _as_action(default, "default")
    positions = {name: index for index, name in enumerate(target, start=1)}

    mapping = {NOOP_ACTION: NOOP_ACTION}

    for index, name in enumerate(source, start=1):
        mapping[index] = positions.get(name, fill)

    return mapping


def _as_array(actions: Any, name: str = "actions") -> np.ndarray:
    """Return actions as an array, coercing an int or a sequence of them into one.

    This shares a name with ObservationPreprocessor's helper but has the opposite
    contract: a frame is always already an array, whereas an action is usually an int.
    """
    if actions is None:
        raise TypeError(
            f"{name} must be an action index, a sequence of them or a numpy array, got "
            "None."
        )

    if isinstance(actions, np.ndarray):
        return actions

    try:
        return np.asarray(actions)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be an action index, a sequence of them or a numpy array, got "
            f"{type(actions).__name__}."
        ) from error


def _check_index_dtype(array: np.ndarray, name: str) -> None:
    """Refuse an array whose values could not be action indices."""
    if array.dtype == np.bool_:
        raise TypeError(
            f"{name} must hold action indices, got a boolean array; True would silently "
            "become action 1."
        )

    if np.issubdtype(array.dtype, np.floating):
        raise TypeError(
            f"{name} must hold integer action indices, got dtype {array.dtype}; an "
            "encoded action belongs in to_index."
        )

    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"{name} must hold integer action indices, got dtype {array.dtype}."
        )


def _as_indices(actions: Any, name: str = "actions") -> np.ndarray:
    """Return action indices as int64, of rank 0, 1 or 2."""
    array = _as_array(actions, name)

    _check_index_dtype(array, name)

    if array.ndim > 2:
        raise ValueError(
            f"{name} must be one action (), an episode (N,) or a history block (N, K), "
            f"got shape {array.shape}."
        )

    # astype copies, so no method can reach back into the caller's recorded actions.
    indices = array.astype(INDEX_DTYPE)

    if indices.size and int(indices.min()) < 0:
        raise ValueError(f"{name} must not be negative, got {int(indices.min())}.")

    return indices


def _as_episode(actions: Any, name: str = "actions") -> np.ndarray:
    """Return a time-ordered episode of action indices, which is always rank 1."""
    indices = _as_indices(actions, name)

    if indices.ndim != 1:
        raise ValueError(
            f"{name} must be an episode of shape (N,) - one action has no history and a "
            f"history block has no single time axis - got shape {indices.shape}."
        )

    return indices


def _as_encoded(encoded: Any, name: str = "encoded") -> np.ndarray:
    """Return an encoded action, whose last axis is always the class axis."""
    array = _as_array(encoded, name)

    if array.dtype == np.bool_ or array.dtype.kind in "USO":
        raise TypeError(f"{name} must hold numbers, got dtype {array.dtype}.")

    if array.ndim not in (1, 2, 3):
        raise ValueError(
            f"{name} must be one encoded action (C,), an encoded episode (N, C) or an "
            f"encoded history (N, K, C), got shape {array.shape}."
        )

    if array.shape[-1] == 0:
        raise ValueError(
            f"{name} must carry at least one class, got shape {array.shape}."
        )

    return array


def _is_single(indices: np.ndarray) -> bool:
    """Say whether an index array holds one action rather than an episode."""
    return indices.ndim == 0


def _as_num_classes(num_classes: Any, name: str = "num_classes") -> int:
    """Return a positive class count, refusing anything that is not one."""
    if isinstance(num_classes, (bool, np.bool_)) or not isinstance(
        num_classes, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer, got {type(num_classes).__name__}.")

    count = int(num_classes)

    if count < 1:
        raise ValueError(f"{name} must be at least 1, got {count}.")

    return count


def _as_action(value: Any, name: str) -> int:
    """Return a single non-negative action index, for a noop or a fill value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an action index, got {type(value).__name__}.")

    action = int(value)

    if action < 0:
        raise ValueError(f"{name} must not be negative, got {action}.")

    return action


def _as_names(action_names: Any, name: str = "action_names") -> tuple:
    """Return a key list as a tuple of unique names, refusing a missing one."""
    if action_names is None:
        raise TypeError(
            f"{name} must be the key list that produced the indices, got None; a "
            "trajectory loaded from a store carries it as trajectory.action_names."
        )

    if isinstance(action_names, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence of key names, got one "
            f"{type(action_names).__name__}."
        )

    names = tuple(action_names)

    for index, entry in enumerate(names):
        if not isinstance(entry, (str, np.str_)):
            raise TypeError(
                f"{name}[{index}] must be a key name, got {type(entry).__name__}."
            )

        if not entry:
            raise ValueError(f"{name}[{index}] must not be empty.")

    if len(set(names)) != len(names):
        raise ValueError(f"{name} must not repeat a key, got {list(names)}.")

    return tuple(str(entry) for entry in names)


def _labels(action_names: Any, noop: Any, name: str = "action_names") -> tuple:
    """Return the label of every action index, noop first, as playback names them."""
    names = _as_names(action_names, name)

    if not isinstance(noop, (str, np.str_)):
        raise TypeError(f"noop must be a name, got {type(noop).__name__}.")

    label = str(noop)

    if label in names:
        raise ValueError(
            f"noop name {label!r} is also a key in {name}; pass a different noop."
        )

    return (label,) + names


def _as_mapping(mapping: Any, name: str = "mapping") -> Dict[int, int]:
    """Return a mapping of action index onto action index, refusing anything else."""
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"{name} must be a mapping of index to index, got "
            f"{type(mapping).__name__}."
        )

    if not mapping:
        raise ValueError(f"{name} must hold at least one entry, got an empty mapping.")

    return {
        _as_action(source, f"{name} key"): _as_action(target, f"{name}[{source!r}]")
        for source, target in mapping.items()
    }


def _lookup_table(mapping: Dict[int, int], default: int, size: int) -> np.ndarray:
    """Build the int64 table that remaps a whole episode in one gather."""
    table = np.full(size, default, dtype=INDEX_DTYPE)

    for source, target in mapping.items():
        table[source] = target

    return table


def _check_range(indices: np.ndarray, num_classes: int, name: str = "actions") -> None:
    """Refuse an action index no policy of this size could ever produce."""
    if indices.size and int(indices.max()) >= num_classes:
        raise ValueError(
            f"{name} must be below num_classes {num_classes}, got {int(indices.max())}; "
            "num_actions_from_names counts the noop that meta.json leaves out."
        )


if __name__ == "__main__":
    # Example usage: run one training-time pipeline over a synthetic episode shaped like a
    # real recording - mostly noop, with the key held in short bursts - and over a single
    # captured action, which is the same code either way.
    from functools import partial

    rng = np.random.default_rng(0)
    names = ("space",)
    classes = num_actions_from_names(names)

    episode = np.zeros(77, dtype=np.int64)

    for start in rng.choice(70, size=4, replace=False):
        episode[start : start + 2] = 1

    stages = [
        ("captured", lambda actions: actions),
        ("hold", partial(ActionPreprocessor.hold, max_hold=3)),
        ("action_history", partial(ActionPreprocessor.action_history, length=4)),
        ("to_onehot", partial(ActionPreprocessor.to_onehot, num_classes=classes)),
        ("flatten_history", ActionPreprocessor.flatten_history),
    ]

    print("--- an episode")
    actions = episode

    for label, stage in stages:
        actions = stage(actions)
        print(f"{label:16s} {str(actions.shape):20s} {actions.dtype}")

    print("--- one action")
    single = 1

    for label, stage in [
        ("captured", ActionPreprocessor.add_batch_dim),
        ("to_onehot", partial(ActionPreprocessor.to_onehot, num_classes=classes)),
    ]:
        single = stage(single)
        print(f"{label:16s} {str(single.shape):20s} {single.dtype}")

    print("--- vocabulary")
    # Show a window around the first press, since a real recording is mostly noop.
    first = int(np.argmax(episode != 0))
    window = episode[max(first - 3, 0) : first + 6]
    print("num_actions    ", classes)
    print("names          ", ActionPreprocessor.to_names(window, names).tolist())

    pipeline = ActionPreprocessor.compose(*(stage for _, stage in stages))
    print("composed       ", pipeline(episode).shape, pipeline(episode).dtype)
