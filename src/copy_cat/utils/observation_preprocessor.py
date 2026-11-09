from collections import deque
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

# ITU-R BT.601 luma weights, in RGB order because that is how frames are captured.
GRAYSCALE_WEIGHTS = (0.299, 0.587, 0.114)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ObservationPreprocessor:
    """A library of static, cascadable transforms for captured observations.

    Every method takes a numpy array and returns a numpy array, so they chain freely:
    the output of one is a valid input to the next. Frames arrive from
    ScreenObservationCapture as RGB uint8 of shape (H, W, 3) - a full screen is far too
    large to store or to feed a policy - so preprocessing is usually applied inside the
    recording loop, where it also shrinks what TrajectoryStore writes to disk.

    Every method also accepts a whole episode at once, so a recording can be preprocessed
    in one call instead of a frame at a time. Rank alone says which is which:

        (H, W)          one grayscale frame
        (H, W, C)       one channelled frame
        (N, H, W, C)    a batch of N frames

    Nothing is inferred from axis sizes, so nothing can be guessed wrong. The cost of that
    is that a batch always carries a channel axis: to_grayscale on a batch returns
    (N, H, W, 1) rather than (N, H, W), keeping the batch 4-D so every later stage stays
    unambiguous.

    Nothing here mutates its argument. Observations loaded from a store are memory-mapped
    and read-only, so every method builds a new array instead of writing in place.

    Preprocessors that take arguments are bound with functools.partial and assembled with
    compose:

        from functools import partial
        from copy_cat.utils.observation_preprocessor import ObservationPreprocessor as OP

        pipeline = OP.compose(
            OP.to_grayscale,
            partial(OP.resize, size=(84, 84)),
            OP.normalize,
        )
        observation = pipeline(frame)      # (900, 1440, 3) uint8 -> (84, 84) float32
        episode = pipeline(observations)   # (77, 900, 1440, 3)   -> (77, 84, 84, 1)

    Only resize needs OpenCV, which is the optional [replay] extra; everything else is
    pure numpy and works on a base install. stack_frames is the one exception to the
    array-in rule, since it joins a sequence of separate frames.
    """

    # ------------------------------------------------------------------ colour

    @staticmethod
    def to_grayscale(obs: np.ndarray, keepdims: bool = False) -> np.ndarray:
        """Collapse an RGB frame to luminance, keeping the input's dtype.

        An alpha channel is dropped first, and an already-grayscale frame passes through.
        Integer frames are rounded back to their own dtype so the result can still be
        stored and replayed; float frames stay float. A batch always keeps its channel
        axis, so keepdims applies to a single frame only.
        """
        frame = _as_observation(obs)
        channels = _channels(frame)

        if channels is None:
            gray = frame
        elif channels == 1:
            gray = frame[..., 0]
        elif channels in (3, 4):
            weights = np.asarray(GRAYSCALE_WEIGHTS, dtype=np.float32)
            luma = frame[..., :3].astype(np.float32) @ weights
            gray = _to_dtype(luma, frame.dtype)
        else:
            raise ValueError(
                f"to_grayscale needs a 1, 3 or 4 channel frame, got {channels} channels."
            )

        return gray[..., np.newaxis] if keepdims or _is_batch(frame) else gray

    @staticmethod
    def to_rgb(obs: np.ndarray) -> np.ndarray:
        """Expand a grayscale frame to three channels, for a 3-channel network."""
        frame = _as_observation(obs)
        channels = _channels(frame)

        if channels == 3:
            return frame

        if channels is None:
            return np.repeat(frame[..., np.newaxis], 3, axis=-1)

        if channels == 1:
            return np.repeat(frame, 3, axis=-1)

        if channels == 4:
            return np.ascontiguousarray(frame[..., :3])

        raise ValueError(
            f"to_rgb needs a 1, 3 or 4 channel frame, got {channels} channels."
        )

    @staticmethod
    def swap_channels(obs: np.ndarray) -> np.ndarray:
        """Reverse the colour channels, converting between RGB and BGR.

        Frames are captured and stored RGB, so this is what OpenCV wants. An alpha
        channel stays where it is; a grayscale frame passes through.
        """
        frame = _as_observation(obs)
        channels = _channels(frame)

        if channels is None or channels == 1:
            return frame

        if channels == 3:
            return np.ascontiguousarray(frame[..., ::-1])

        if channels == 4:
            return np.ascontiguousarray(
                np.concatenate([frame[..., 2::-1], frame[..., 3:]], axis=-1)
            )

        raise ValueError(
            f"swap_channels needs a 1, 3 or 4 channel frame, got {channels} channels."
        )

    @staticmethod
    def drop_alpha(obs: np.ndarray) -> np.ndarray:
        """Discard a fourth channel, turning BGRA or RGBA into three channels."""
        frame = _as_observation(obs)
        channels = _channels(frame)

        if channels is None or channels <= 3:
            return frame

        return np.ascontiguousarray(frame[..., :3])

    # ---------------------------------------------------------------- geometry

    @staticmethod
    def resize(
        obs: np.ndarray,
        size: Tuple[int, int],
        interpolation: str = "area",
    ) -> np.ndarray:
        """Resize a frame to size, given as (height, width).

        This is the one preprocessor that needs OpenCV, because good interpolation is
        not worth reimplementing. Use downsample for a dependency-free shrink. The
        default "area" is what you want when scaling a screen capture down.
        """
        frame = _as_observation(obs)
        height, width = _as_size(size)
        channels = _channels(frame)
        cv2 = _import_cv2()

        modes = {
            "area": cv2.INTER_AREA,
            "linear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "cubic": cv2.INTER_CUBIC,
        }

        if interpolation not in modes:
            raise ValueError(
                f"interpolation must be one of {sorted(modes)}, got {interpolation!r}."
            )

        if channels is not None and channels > 4:
            raise ValueError(
                f"OpenCV cannot resize a frame of {channels} channels; resize "
                "before stacking."
            )

        def resize_one(image: np.ndarray) -> np.ndarray:
            """Resize a single frame, which is all OpenCV handles."""
            # OpenCV takes (width, height), and needs a writable, contiguous buffer.
            resized = cv2.resize(
                np.ascontiguousarray(image),
                (width, height),
                interpolation=modes[interpolation],
            )

            # OpenCV drops a trailing axis of length 1; put it back so shapes stay stable.
            if image.ndim == 3 and resized.ndim == 2:
                resized = resized[:, :, np.newaxis]

            return resized

        if _is_batch(frame):
            return np.stack([resize_one(image) for image in frame])

        return resize_one(frame)

    @staticmethod
    def crop(obs: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
        """Cut a rectangle out of a frame, to keep only a region of interest."""
        frame = _as_observation(obs)

        if height <= 0 or width <= 0:
            raise ValueError(
                f"Crop height and width must be positive, got {(height, width)}."
            )

        if top < 0 or left < 0:
            raise ValueError(f"Crop top and left must be non-negative, got {(top, left)}.")

        frame_height, frame_width = _spatial(frame)

        if top + height > frame_height or left + width > frame_width:
            raise ValueError(
                f"Crop box {(top, left, height, width)} does not fit a "
                f"{frame_height}x{frame_width} frame."
            )

        window = _window(frame, slice(top, top + height), slice(left, left + width))

        return np.ascontiguousarray(frame[window])

    @staticmethod
    def center_crop(obs: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Cut a centred rectangle of size (height, width) out of a frame."""
        frame = _as_observation(obs)
        height, width = _as_size(size)
        frame_height, frame_width = _spatial(frame)

        if height > frame_height or width > frame_width:
            raise ValueError(
                f"Cannot center-crop {(height, width)} out of a "
                f"{frame_height}x{frame_width} frame."
            )

        top = (frame_height - height) // 2
        left = (frame_width - width) // 2

        return ObservationPreprocessor.crop(frame, top, left, height, width)

    @staticmethod
    def downsample(obs: np.ndarray, factor: int, mode: str = "subsample") -> np.ndarray:
        """Shrink a frame by an integer factor, without needing OpenCV.

        "subsample" keeps every factor-th pixel, which is fast but aliases; "mean"
        averages each block of pixels, dropping the trailing rows and columns that do
        not fill one.
        """
        frame = _as_observation(obs)
        factor = int(factor)

        if factor < 1:
            raise ValueError(f"factor must be at least 1, got {factor}.")

        if factor == 1:
            return frame

        if mode == "subsample":
            window = _window(frame, slice(None, None, factor), slice(None, None, factor))

            return np.ascontiguousarray(frame[window])

        if mode != "mean":
            raise ValueError(f"mode must be 'subsample' or 'mean', got {mode!r}.")

        frame_height, frame_width = _spatial(frame)
        height = (frame_height // factor) * factor
        width = (frame_width // factor) * factor

        if height == 0 or width == 0:
            raise ValueError(
                f"factor {factor} is larger than the {frame_height}x{frame_width} frame."
            )

        lead = _lead(frame)
        trimmed = frame[_window(frame, slice(height), slice(width))]
        blocks = trimmed.reshape(
            *trimmed.shape[:lead],
            height // factor,
            factor,
            width // factor,
            factor,
            *trimmed.shape[lead + 2 :],
        )

        return _to_dtype(
            blocks.mean(axis=(lead + 1, lead + 3), dtype=np.float64), frame.dtype
        )

    @staticmethod
    def pad(
        obs: np.ndarray,
        size: Tuple[int, int],
        value: float = 0,
        center: bool = True,
    ) -> np.ndarray:
        """Grow a frame to size, given as (height, width), filling with value."""
        frame = _as_observation(obs)
        height, width = _as_size(size)
        frame_height, frame_width = _spatial(frame)

        if height < frame_height or width < frame_width:
            raise ValueError(
                f"pad cannot shrink a {frame_height}x{frame_width} frame to "
                f"{(height, width)}."
            )

        extra_rows = height - frame_height
        extra_cols = width - frame_width
        top = extra_rows // 2 if center else 0
        left = extra_cols // 2 if center else 0
        lead = _lead(frame)

        padding = [(0, 0)] * lead
        padding.append((top, extra_rows - top))
        padding.append((left, extra_cols - left))
        padding.extend([(0, 0)] * (frame.ndim - lead - 2))

        return np.pad(frame, padding, mode="constant", constant_values=value)

    @staticmethod
    def flip_horizontal(obs: np.ndarray) -> np.ndarray:
        """Mirror a frame left to right.

        This is augmentation rather than preprocessing, and it silently invalidates the
        left and right actions of a demonstration, so mirror the actions too or leave it
        out of a recording pipeline.
        """
        frame = _as_observation(obs)
        window = _window(frame, slice(None), slice(None, None, -1))

        return np.ascontiguousarray(frame[window])

    @staticmethod
    def flip_vertical(obs: np.ndarray) -> np.ndarray:
        """Mirror a frame top to bottom, as augmentation."""
        frame = _as_observation(obs)
        window = _window(frame, slice(None, None, -1), slice(None))

        return np.ascontiguousarray(frame[window])

    # ------------------------------------------------------------------ values

    @staticmethod
    def normalize(obs: np.ndarray, max_value: Optional[float] = None) -> np.ndarray:
        """Scale a frame into float32 [0, 1], the usual last step before a network.

        max_value defaults to the largest value the frame's dtype holds - 255 for uint8 -
        and to 1.0 for a float frame, which is then only a cast.
        """
        frame = _as_observation(obs)
        limit = _dtype_max(frame.dtype) if max_value is None else float(max_value)

        if limit <= 0:
            raise ValueError(f"max_value must be positive, got {limit}.")

        return frame.astype(np.float32) / np.float32(limit)

    @staticmethod
    def standardize(
        obs: np.ndarray,
        mean: Union[float, Sequence[float]] = IMAGENET_MEAN,
        std: Union[float, Sequence[float]] = IMAGENET_STD,
    ) -> np.ndarray:
        """Centre and scale a frame per channel, as pretrained backbones expect.

        The defaults are the ImageNet statistics, which assume the frame is already in
        [0, 1], so this cascades after normalize. Pass scalars for a grayscale frame.
        """
        frame = _as_observation(obs)
        channels = _channels(frame)
        centre = _as_stats(mean, channels, "mean")
        spread = _as_stats(std, channels, "std")

        if np.any(spread == 0):
            raise ValueError(f"std must be non-zero, got {spread.tolist()}.")

        return (frame.astype(np.float32) - centre) / spread

    @staticmethod
    def scale_to_range(
        obs: np.ndarray,
        low: float = -1.0,
        high: float = 1.0,
        max_value: Optional[float] = None,
    ) -> np.ndarray:
        """Scale a frame into float32 [low, high], for a tanh-shaped input range."""
        if low >= high:
            raise ValueError(f"low must be below high, got {(low, high)}.")

        unit = ObservationPreprocessor.normalize(obs, max_value=max_value)

        return unit * np.float32(high - low) + np.float32(low)

    @staticmethod
    def to_uint8(obs: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        """Bring a frame back to uint8, so a processed observation stays viewable.

        This undoes normalize: a float frame is read as spanning [low, high]. An integer
        frame is only clipped and cast.
        """
        frame = _as_observation(obs)

        if np.issubdtype(frame.dtype, np.integer):
            return np.clip(frame, 0, 255).astype(np.uint8)

        if low >= high:
            raise ValueError(f"low must be below high, got {(low, high)}.")

        scaled = (frame.astype(np.float32) - np.float32(low)) / np.float32(high - low)

        return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)

    @staticmethod
    def clip(obs: np.ndarray, low: float, high: float) -> np.ndarray:
        """Limit a frame's values to [low, high], keeping its dtype."""
        frame = _as_observation(obs)

        if low >= high:
            raise ValueError(f"low must be below high, got {(low, high)}.")

        return np.clip(frame, low, high).astype(frame.dtype, copy=False)

    @staticmethod
    def binarize(
        obs: np.ndarray,
        threshold: float,
        high: Optional[float] = None,
    ) -> np.ndarray:
        """Turn a frame into a two-valued mask, keeping its dtype.

        Pixels above threshold become high, which defaults to the dtype's largest value,
        and the rest become zero.
        """
        frame = _as_observation(obs)
        top = _dtype_max(frame.dtype) if high is None else high

        return np.where(frame > threshold, frame.dtype.type(top), frame.dtype.type(0))

    @staticmethod
    def quantize(
        obs: np.ndarray,
        levels: int,
        max_value: Optional[float] = None,
    ) -> np.ndarray:
        """Round a frame onto evenly spaced levels, reducing its colour depth.

        Fewer distinct values make a screen frame far more compressible, and small enough
        counts make a tabular agent possible on a simple game.
        """
        frame = _as_observation(obs)
        levels = int(levels)

        if levels < 2:
            raise ValueError(f"levels must be at least 2, got {levels}.")

        limit = _dtype_max(frame.dtype) if max_value is None else float(max_value)

        if limit <= 0:
            raise ValueError(f"max_value must be positive, got {limit}.")

        step = limit / (levels - 1)
        buckets = np.clip(np.rint(frame.astype(np.float64) / step), 0, levels - 1)

        return _to_dtype(buckets * step, frame.dtype)

    # ------------------------------------------------------------------ layout

    @staticmethod
    def to_channels_first(obs: np.ndarray) -> np.ndarray:
        """Turn a (H, W, C) frame into (C, H, W), the layout torch models expect.

        A grayscale (H, W) frame gains a leading channel axis of one, and a batch becomes
        (N, C, H, W). The result no longer matches this class's channels-last convention,
        so this belongs at the end of a pipeline.
        """
        frame = _as_observation(obs)

        if _is_batch(frame):
            return np.ascontiguousarray(np.moveaxis(frame, -1, 1))

        if frame.ndim == 2:
            return frame[np.newaxis, :, :]

        return np.ascontiguousarray(np.transpose(frame, (2, 0, 1)))

    @staticmethod
    def to_channels_last(obs: np.ndarray) -> np.ndarray:
        """Turn a (C, H, W) frame into (H, W, C), the layout everything else expects.

        A grayscale (H, W) frame gains a trailing channel axis, and a (N, C, H, W) batch
        becomes (N, H, W, C).
        """
        frame = _as_observation(obs)

        if _is_batch(frame):
            return np.ascontiguousarray(np.moveaxis(frame, 1, -1))

        if frame.ndim == 2:
            return frame[:, :, np.newaxis]

        return np.ascontiguousarray(np.transpose(frame, (1, 2, 0)))

    @staticmethod
    def add_batch_dim(obs: np.ndarray) -> np.ndarray:
        """Prepend an axis of one, so a single frame looks like a batch of one.

        A batch is returned unchanged, so this can be used to guarantee a batched shape.
        """
        frame = _as_observation(obs)

        if _is_batch(frame):
            return frame

        return frame[np.newaxis, ...]

    @staticmethod
    def flatten(obs: np.ndarray) -> np.ndarray:
        """Collapse a frame to one dimension, for an MLP policy or a state key.

        A batch is flattened per frame instead, to one row of features each.
        """
        frame = _as_observation(obs)

        if _is_batch(frame):
            return frame.reshape(len(frame), -1)

        return np.ravel(frame)

    # ---------------------------------------------------------------- temporal

    @staticmethod
    def framestack(
        obs: np.ndarray,
        num_frames: int = 4,
        fill: str = "repeat",
    ) -> np.ndarray:
        """Stack each frame together with the frames before it, so motion is visible.

        This is the frame stack that lets an agent read velocity off still images. Given a
        batch it slides a window over time, so frame i carries frames i - num_frames + 1
        through i on the channel axis, oldest first, and the batch keeps its length. The
        first frames have no history: fill="repeat" pads with the earliest frame, and
        fill="zeros" pads with black. A single frame has no history at all, so it is
        simply repeated.

        The gather holds num_frames copies of the batch, so stack late in a pipeline -
        after downsampling - rather than over full-resolution frames.
        """
        frame = _as_observation(obs)
        num_frames = int(num_frames)

        if num_frames < 1:
            raise ValueError(f"num_frames must be at least 1, got {num_frames}.")

        if fill not in {"repeat", "zeros"}:
            raise ValueError(f"fill must be 'repeat' or 'zeros', got {fill!r}.")

        if not _is_batch(frame):
            if fill == "repeat":
                history = [frame] * num_frames
            else:
                history = [np.zeros_like(frame)] * (num_frames - 1) + [frame]

            return ObservationPreprocessor.stack_frames(history)

        count, height, width, channels = frame.shape
        offsets = np.arange(-num_frames + 1, 1)
        indices = np.arange(count)[:, np.newaxis] + offsets
        taken = frame[np.clip(indices, 0, None)]

        if fill == "zeros":
            missing = (indices < 0)[:, :, np.newaxis, np.newaxis, np.newaxis]
            taken = np.where(missing, frame.dtype.type(0), taken)

        # (N, S, H, W, C) -> (N, H, W, S * C), which orders the channels oldest frame
        # first, exactly as concatenating the frames would.
        return np.moveaxis(taken, 1, -2).reshape(
            count, height, width, num_frames * channels
        )

    @staticmethod
    def framestack_stream(
        num_frames: int = 4,
        axis: int = -1,
        fill: str = "repeat",
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return a stateful framestack for frames that arrive one at a time.

        Use this while recording, where the episode does not exist yet; use framestack
        offline, once it does. The first output is bootstrapped either by repeating the
        first frame (fill="repeat") or by left-padding with zeros (fill="zeros"), and
        the returned stage carries a reset() for starting a new episode.
        """
        num_frames = int(num_frames)

        if num_frames < 1:
            raise ValueError(f"num_frames must be at least 1, got {num_frames}.")

        if fill not in {"repeat", "zeros"}:
            raise ValueError(f"fill must be 'repeat' or 'zeros', got {fill!r}.")

        frames = deque(maxlen=num_frames)

        def stage(obs: np.ndarray) -> np.ndarray:
            frame = _as_observation(obs)

            if _is_batch(frame):
                raise ValueError(
                    "framestack_stream takes one frame at a time; use framestack for a "
                    f"batch, got shape {frame.shape}."
                )

            if not frames:
                if fill == "repeat":
                    frames.extend(frame.copy() for _ in range(num_frames))
                else:
                    frames.extend(np.zeros_like(frame) for _ in range(num_frames - 1))
                    frames.append(frame.copy())
            else:
                first = frames[0]

                if frame.shape != first.shape or frame.dtype != first.dtype:
                    raise ValueError(
                        f"All frames must share one shape and dtype. Initial frame is "
                        f"{first.shape}/{first.dtype} but current frame is "
                        f"{frame.shape}/{frame.dtype}."
                    )

                frames.append(frame.copy())

            return ObservationPreprocessor.stack_frames(tuple(frames), axis=axis)

        def reset() -> None:
            """Clear the internal stack before a new episode."""
            frames.clear()

        stage.reset = reset

        return stage

    @staticmethod
    def stack_frames(frames: Sequence[np.ndarray], axis: int = -1) -> np.ndarray:
        """Join several separate frames into one observation.

        Unlike the rest of the class this takes a sequence rather than one array, for
        frames that are not already batched together. Grayscale frames gain a new axis;
        channelled frames are concatenated along axis.
        """
        stack = list(frames)

        if not stack:
            raise ValueError("stack_frames needs at least one frame, got none.")

        first = _as_observation(stack[0], "frames[0]")

        for index, frame in enumerate(stack):
            frame = _as_observation(frame, f"frames[{index}]")

            if frame.shape != first.shape or frame.dtype != first.dtype:
                raise ValueError(
                    f"All frames must share one shape and dtype. frames[0] is "
                    f"{first.shape}/{first.dtype} but frames[{index}] is "
                    f"{frame.shape}/{frame.dtype}."
                )

        if first.ndim == 2:
            return np.stack(stack, axis=axis)

        return np.concatenate(stack, axis=axis)

    @staticmethod
    def frame_difference(
        current: np.ndarray,
        previous: np.ndarray,
        signed: bool = False,
    ) -> np.ndarray:
        """Return the per-pixel change between two frames, highlighting what moved.

        By default this is the absolute difference in the frames' own dtype; signed keeps
        the direction of the change, in a wider dtype that can hold it.
        """
        frame = _as_observation(current, "current")
        earlier = _as_observation(previous, "previous")

        if frame.shape != earlier.shape:
            raise ValueError(
                f"current and previous must have the same shape, got {frame.shape} "
                f"and {earlier.shape}."
            )

        if np.issubdtype(frame.dtype, np.integer):
            # Subtracting in a signed dtype, so a uint8 difference cannot wrap around.
            working = np.int16 if frame.dtype.itemsize <= 1 else np.int64
        else:
            working = np.float32

        difference = frame.astype(working) - earlier.astype(working)

        if signed:
            return difference

        return _to_dtype(np.abs(difference), frame.dtype)

    # ---------------------------------------------------------------- plumbing

    @staticmethod
    def compose(*preprocessors: Callable[[np.ndarray], np.ndarray]) -> Callable:
        """Chain preprocessors into one callable, applied left to right.

        Bind arguments with functools.partial, so the pipeline is built once and reused
        for every frame of a recording.
        """
        for index, preprocessor in enumerate(preprocessors):
            if not callable(preprocessor):
                raise ValueError(
                    f"compose takes callables; argument {index} is a "
                    f"{type(preprocessor).__name__}."
                )

        def pipeline(obs: np.ndarray) -> np.ndarray:
            """Apply every composed preprocessor to an observation, in order."""
            for preprocessor in preprocessors:
                obs = preprocessor(obs)

            return obs

        return pipeline


def _as_array(obs: Any, name: str = "obs") -> np.ndarray:
    """Return an observation, refusing anything that is not a numpy array."""
    if not isinstance(obs, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(obs).__name__}.")

    return obs


def _as_observation(obs: Any, name: str = "obs") -> np.ndarray:
    """Return an observation, refusing anything that is not a frame or a batch of them."""
    frame = _as_array(obs, name)

    if frame.ndim not in (2, 3, 4):
        raise ValueError(
            f"{name} must be a frame - (H, W) or (H, W, C) - or a batch of frames "
            f"(N, H, W, C), got shape {frame.shape}."
        )

    return frame


def _is_batch(frame: np.ndarray) -> bool:
    """Say whether an observation holds several frames rather than one."""
    return frame.ndim == 4


def _lead(frame: np.ndarray) -> int:
    """Return how many axes come before a frame's own, which is 1 for a batch."""
    return 1 if _is_batch(frame) else 0


def _channels(frame: np.ndarray) -> Optional[int]:
    """Return a frame's channel count, or None if it carries no channel axis."""
    return None if frame.ndim == 2 else frame.shape[-1]


def _spatial(frame: np.ndarray) -> Tuple[int, int]:
    """Return a frame's (height, width), which always follow the batch axis."""
    lead = _lead(frame)

    return frame.shape[lead], frame.shape[lead + 1]


def _window(frame: np.ndarray, rows: slice, cols: slice) -> Tuple[Any, ...]:
    """Build an index that slices a frame's height and width, batched or not."""
    return (slice(None),) * _lead(frame) + (rows, cols)


def _dtype_max(dtype: np.dtype) -> float:
    """Return the largest value a frame of this dtype is expected to hold."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)

    if np.issubdtype(dtype, np.floating):
        return 1.0

    raise TypeError(f"Observations must be integer or float arrays, got {dtype}.")


def _to_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a computed float result back to a frame dtype, rounding integers."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)

        return np.clip(np.rint(values), info.min, info.max).astype(dtype)

    return values.astype(dtype)


def _as_size(size: Any, name: str = "size") -> Tuple[int, int]:
    """Return a (height, width) pair, refusing anything that is not one."""
    try:
        height, width = size
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a (height, width) pair, got {size!r}."
        ) from None

    height, width = int(height), int(width)

    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {(height, width)}.")

    return height, width


def _as_stats(
    value: Union[float, Sequence[float]], channels: Optional[int], name: str
) -> np.ndarray:
    """Return per-channel statistics that broadcast against a frame."""
    stats = np.asarray(value, dtype=np.float32).reshape(-1)

    if stats.size == 0:
        raise ValueError(f"{name} must hold at least one value, got {value!r}.")

    if stats.size == 1:
        return stats

    if channels is None:
        raise ValueError(
            f"{name} has {stats.size} values but the frame is grayscale; pass a scalar."
        )

    if stats.size != channels:
        raise ValueError(
            f"{name} must have one value per channel; got {stats.size} for "
            f"{channels} channels."
        )

    return stats


def _import_cv2():
    """Import OpenCV, which is only needed for resize."""
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "Resizing an observation needs OpenCV, which is an optional dependency. "
            "Install it with: pip install 'copy-cat[replay]', or use downsample instead."
        ) from error

    return cv2


if __name__ == "__main__":
    # Example usage: run one recording-time pipeline over a single synthetic screen frame
    # and over a whole synthetic episode, which is the same code either way.
    from functools import partial

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(900, 1440, 3), dtype=np.uint8)
    episode = rng.integers(0, 255, size=(5, 900, 1440, 3), dtype=np.uint8)

    # downsample rather than resize, so the example runs without OpenCV.
    stages = [
        ("captured", lambda obs: obs),
        ("to_grayscale", ObservationPreprocessor.to_grayscale),
        ("downsample", partial(ObservationPreprocessor.downsample, factor=4, mode="mean")),
        ("center_crop", partial(ObservationPreprocessor.center_crop, size=(84, 84))),
        ("framestack", partial(ObservationPreprocessor.framestack, num_frames=4)),
        ("normalize", ObservationPreprocessor.normalize),
    ]

    for title, observations in (("one frame", frame), ("an episode", episode)):
        print(f"--- {title}")
        observation = observations

        for label, stage in stages:
            observation = stage(observation)
            print(f"{label:14s} {str(observation.shape):20s} {observation.dtype}")

    pipeline = ObservationPreprocessor.compose(*(stage for _, stage in stages))
    print("composed       ", pipeline(frame).shape, pipeline(episode).shape)
