import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from copy_cat.preprocessing.observation_preprocessor import ObservationPreprocessor
from copy_cat.data.trajectory import Trajectory

META_FILE = "meta.json"
STEPS_FILE = "steps.jsonl"
OBSERVATIONS_FILE = "observations.npy"

# Used for playback only, when an episode carries neither timestamps nor a recorded rate.
DEFAULT_FPS = 10.0

# Playback only: the grey line drawn between the frames of a stacked observation.
TILE_SEPARATOR = 2
TILE_SEPARATOR_VALUE = 128


def frame_delays(
    timestamps: Sequence[Optional[float]],
    fps: Optional[float] = None,
    speed: float = 1.0,
    fallback_fps: float = DEFAULT_FPS,
) -> List[float]:
    """Return how long each frame should stay on screen, in seconds.

    Recorded timestamps are preferred over a constant rate because stripping no-op
    steps leaves a trajectory unevenly sampled in time, and only the timestamps
    still carry those gaps. Passing fps overrides them and plays at a fixed rate;
    fallback_fps applies when the timestamps are missing or unusable.
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}.")

    count = len(timestamps)

    if count == 0:
        return []

    if fps is not None and fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    if fallback_fps <= 0:
        raise ValueError(f"fallback_fps must be positive, got {fallback_fps}.")

    usable = fps is None and count >= 2 and all(_is_number(stamp) for stamp in timestamps)

    if not usable:
        constant = 1.0 / (fps if fps is not None else fallback_fps)

        return [constant / speed] * count

    # Each frame is held until the next one was recorded; the last inherits the gap
    # before it, since nothing says how long it stayed on screen.
    gaps = [
        max(0.0, float(timestamps[i + 1]) - float(timestamps[i])) for i in range(count - 1)
    ]
    gaps.append(gaps[-1])

    return [gap / speed for gap in gaps]


class TrajectoryStore:
    """Stores demonstrations on disk as a directory of named episodes.

    Each episode is a directory holding meta.json, steps.jsonl and observations.npy,
    so the observations stay a plain numpy array that any tool can open, and can be
    memory-mapped instead of read into RAM. Screen recordings are large - a minute at
    10 Hz of 1080p is several gigabytes - so neither saving nor loading ever holds a
    second copy of the frame stack.

    Episodes are written to a temporary directory and moved into place, so an
    interrupted save leaves no half-written episode behind.
    """

    def __init__(self, root: Union[str, Path], create: bool = True):
        self.root = Path(root).expanduser()

        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir():
            raise FileNotFoundError(f"Store root {self.root} does not exist.")

    def save(
        self,
        trajectory: Trajectory,
        name: Optional[str] = None,
        fps: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> str:
        """Write a trajectory to the store and return the episode name."""
        name = self._generate_name() if name is None else _validate_name(name)
        destination = self.root / name

        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Episode {name!r} already exists. Pass overwrite=True to replace it."
            )

        steps = trajectory.get_trajectory_length()
        states = [trajectory.get_state(step) for step in range(steps)]
        timestamps = [
            trajectory.additional_info[step].get("timestamp") for step in range(steps)
        ]

        staging = self.root / f".tmp-{name}-{os.getpid()}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        try:
            # Steps first: they are small, so anything unserialisable fails here
            # rather than after several gigabytes of frames have been written.
            _write_steps(staging / STEPS_FILE, trajectory)
            shape, dtype = _write_observations(staging / OBSERVATIONS_FILE, states)

            meta = {
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "num_steps": steps,
                "fps": fps if fps is not None else _infer_fps(timestamps),
                "obs_shape": list(shape) if shape is not None else None,
                "obs_dtype": dtype,
                "action_names": _jsonable(getattr(trajectory, "action_names", None)),
                "metadata": _jsonable(metadata) if metadata is not None else {},
            }
            _write_json(staging / META_FILE, meta)

            # Replacing a directory needs the old one gone first, so the previous
            # episode only disappears once the new one is fully written.
            if destination.exists():
                shutil.rmtree(destination)

            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        return name

    def load(self, name: str, mmap: bool = True) -> Trajectory:
        """Load an episode back into a Trajectory.

        Observations are memory-mapped by default, so opening a multi-gigabyte
        episode is instant and the frames are paged in as they are touched.
        """
        path = self._require(name)
        steps = _read_steps(path / STEPS_FILE)
        frames = _open_observations(path / OBSERVATIONS_FILE, mmap=mmap)

        if frames is not None and len(frames) != len(steps):
            raise ValueError(
                f"Episode {name!r} is inconsistent: {len(frames)} observations "
                f"but {len(steps)} steps."
            )

        if frames is None and steps:
            raise FileNotFoundError(
                f"Episode {name!r} has {len(steps)} steps but no {OBSERVATIONS_FILE}."
            )

        trajectory = Trajectory()

        for index, step in enumerate(steps):
            trajectory.record_step(
                state=frames[index],
                action=step.get("action"),
                reward=step.get("reward"),
                **(step.get("info") or {}),
            )

        action_names = self.info(name).get("action_names")
        trajectory.action_names = (
            tuple(action_names) if action_names is not None else None
        )

        return trajectory

    def play(
        self,
        name: str,
        fps: Optional[float] = None,
        speed: float = 1.0,
        scale: Optional[float] = None,
        loop: bool = False,
        overlay: bool = True,
        window: Optional[str] = None,
        is_framestacked: bool = False,
    ) -> None:
        """Replay an episode in a window, with the recorded action drawn on each frame.

        Playback follows the recorded timestamps so gaps left by stripped no-op steps
        are preserved; fps forces a constant rate instead, and speed scales either.
        Press q or Esc to quit, space to pause and resume.

        is_framestacked says the observations carry a stack of frames on their last axis -
        (H, W, S) when the frames are grayscale, (H, W, C, S) when they have channels -
        and draws the stack's frames side by side, oldest first, so the history an agent
        sees is visible. It is not inferred, because a stack of three grayscale frames and
        one RGB frame are the same shape. Float observations, such as those a pipeline
        ending in normalize produces, are read as spanning [0, 1] and shown as uint8.
        """
        cv2 = _import_cv2()

        path = self._require(name)
        meta = self.info(name)
        steps = _read_steps(path / STEPS_FILE)

        if not steps:
            print(f"Episode {name!r} has no steps to play.")

            return

        frames = _open_observations(path / OBSERVATIONS_FILE, mmap=True)

        if frames is None:
            raise FileNotFoundError(f"Episode {name!r} has no {OBSERVATIONS_FILE}.")

        timestamps = [(step.get("info") or {}).get("timestamp") for step in steps]
        delays = frame_delays(
            timestamps,
            fps=fps,
            speed=speed,
            fallback_fps=meta.get("fps") or DEFAULT_FPS,
        )

        action_names = meta.get("action_names")
        window = window or f"copy-cat: {name}"

        # WINDOW_NORMAL so a full-screen recording can be resized down to fit.
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        try:
            while True:
                for index, step in enumerate(steps):
                    started = time.perf_counter()
                    frame = _to_display(frames[index], stacked=is_framestacked)

                    if scale is not None and scale != 1.0:
                        frame = cv2.resize(
                            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                        )

                    if overlay:
                        _draw_overlay(
                            cv2,
                            frame,
                            index=index,
                            total=len(steps),
                            timestamp=timestamps[index],
                            action=step.get("action"),
                            action_names=action_names,
                        )

                        if is_framestacked:
                            _draw_stack_labels(cv2, frame, count=frames.shape[-1])

                    cv2.imshow(window, frame)

                    # Decoding and drawing already consumed part of the interval.
                    remaining = delays[index] - (time.perf_counter() - started)

                    if not _wait(cv2, remaining):
                        return

                if not loop:
                    break
        finally:
            cv2.destroyWindow(window)
            # Some backends only tear the window down on the next event loop turn.
            cv2.waitKey(1)

    def list(self) -> List[str]:
        """Return the names of the episodes in the store, sorted."""
        if not self.root.is_dir():
            return []

        names = [
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir()
            and not entry.name.startswith(".")
            and (entry / META_FILE).is_file()
        ]

        return sorted(names)

    def info(self, name: str) -> Dict[str, Any]:
        """Return an episode's metadata without touching its observations."""
        path = self._require(name)

        with open(path / META_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def exists(self, name: str) -> bool:
        """Return whether an episode of this name is stored."""
        return (self.path_for(name) / META_FILE).is_file()

    def delete(self, name: str) -> None:
        """Remove an episode and everything in it."""
        shutil.rmtree(self._require(name))

    def path_for(self, name: str) -> Path:
        """Return the directory an episode lives in, whether or not it exists."""
        return self.root / _validate_name(name)

    def _require(self, name: str) -> Path:
        """Return an existing episode's directory, or raise."""
        path = self.path_for(name)

        if not (path / META_FILE).is_file():
            raise FileNotFoundError(f"No episode named {name!r} in {self.root}.")

        return path

    def _generate_name(self) -> str:
        """Return a timestamped name that is not taken yet."""
        base = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        name = base
        suffix = 2

        while (self.root / name).exists():
            name = f"{base}_{suffix}"
            suffix += 1

        return name

    def __len__(self) -> int:
        """Return the number of stored episodes."""
        return len(self.list())

    def __iter__(self) -> Iterator[str]:
        """Iterate over the stored episode names."""
        return iter(self.list())

    def __contains__(self, name: object) -> bool:
        """Return whether an episode of this name is stored."""
        return isinstance(name, str) and self.exists(name)

    def __repr__(self) -> str:
        """Return a readable description of the store."""
        return f"TrajectoryStore(root={str(self.root)!r}, episodes={len(self)})"


def _write_observations(path: Path, states: Sequence[Any]):
    """Write the observations as a single .npy stack, one frame at a time."""
    if not states:
        return None, None

    first = states[0]

    if not isinstance(first, np.ndarray):
        raise TypeError(
            f"Observations must be numpy arrays, got {type(first).__name__} at step 0."
        )

    shape = first.shape
    dtype = first.dtype

    for index, state in enumerate(states):
        if not isinstance(state, np.ndarray):
            raise TypeError(
                f"Observations must be numpy arrays, got "
                f"{type(state).__name__} at step {index}."
            )

        if state.shape != shape or state.dtype != dtype:
            raise ValueError(
                f"Observations must all share one shape and dtype. Step 0 is "
                f"{shape}/{dtype} but step {index} is {state.shape}/{state.dtype}."
            )

    # Filled through a memmap so the frames are never stacked into a second copy.
    stack = np.lib.format.open_memmap(
        path, mode="w+", dtype=dtype, shape=(len(states), *shape)
    )

    try:
        for index, state in enumerate(states):
            stack[index] = state

        stack.flush()
    finally:
        del stack

    return shape, str(dtype)


def _open_observations(path: Path, mmap: bool):
    """Open a stored observation stack, or return None if the episode has none."""
    if not path.is_file():
        return None

    return np.load(path, mmap_mode="r" if mmap else None)


def _write_steps(path: Path, trajectory: Trajectory) -> None:
    """Write one JSON object per step, holding everything but the observation."""
    with open(path, "w", encoding="utf-8") as handle:
        for step in range(trajectory.get_trajectory_length()):
            record = {
                "step": step,
                "action": _jsonable(trajectory.state_actions[step][1]),
                "reward": _jsonable(trajectory.rewards[step]),
                "info": _jsonable(trajectory.additional_info[step] or {}),
            }
            handle.write(json.dumps(record) + "\n")


def _read_steps(path: Path) -> List[Dict[str, Any]]:
    """Read the per-step records of an episode."""
    steps = []

    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} is corrupt at line {number}: {error}") from error

    return steps


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON document."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _jsonable(value: Any) -> Any:
    """Convert a value into something json can write, or say why it cannot."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}

    raise TypeError(
        f"Cannot store {type(value).__name__} in step metadata; only JSON-compatible "
        "values and numpy scalars or arrays are supported."
    )


def _validate_name(name: str) -> str:
    """Return an episode name, refusing anything that could escape the store root."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Episode names must be non-empty strings, got {name!r}.")

    if name != name.strip():
        raise ValueError(f"Episode name {name!r} has leading or trailing whitespace.")

    # Anything but a bare directory name could point outside the root, and a leading
    # dot would collide with the staging directories used while saving.
    if name.startswith(".") or Path(name).name != name or os.sep in name:
        raise ValueError(f"Episode name {name!r} must be a plain directory name.")

    if os.altsep is not None and os.altsep in name:
        raise ValueError(f"Episode name {name!r} must be a plain directory name.")

    return name


def _infer_fps(timestamps: Sequence[Optional[float]]) -> Optional[float]:
    """Estimate the recording rate from step timestamps, if they allow it.

    Stripped no-op steps make this an underestimate of the poll rate, so it is only
    ever recorded as metadata, never used to reconstruct step times.
    """
    stamps = [float(stamp) for stamp in timestamps if _is_number(stamp)]

    if len(stamps) < 2:
        return None

    span = stamps[-1] - stamps[0]

    if span <= 0:
        return None

    return (len(stamps) - 1) / span


def _is_number(value: Any) -> bool:
    """Return whether a value is a real number, excluding bools."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _import_cv2():
    """Import OpenCV, which is only needed for playback."""
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "Replaying an episode needs OpenCV, which is an optional dependency. "
            "Install it with: pip install 'copy-cat[replay]'"
        ) from error

    return cv2


def _to_display(frame: np.ndarray, stacked: bool = False) -> np.ndarray:
    """Turn a stored observation into an image OpenCV can show.

    A stacked observation becomes one image holding its frames side by side, oldest
    first, so a whole stack is watched in a single window.
    """
    if not stacked:
        return _to_bgr(_to_viewable(frame))

    # Three channels throughout, so tiles of any kind join into one image and the
    # overlay's light-on-dark text draws the same way it does on a plain recording.
    tiles = [
        _to_bgr(ObservationPreprocessor.to_rgb(_to_viewable(tile)))
        for tile in _split_stack(frame)
    ]

    return _tile(tiles)


def _split_stack(frame: np.ndarray) -> List[np.ndarray]:
    """Split a stacked observation into its frames, oldest first.

    The stack always sits on the last axis, so one rule covers a grayscale stack and a
    channelled one alike.
    """
    if frame.ndim not in (3, 4):
        raise ValueError(
            f"A stacked observation must be (H, W, S) or (H, W, C, S), got shape "
            f"{frame.shape}. Pass is_framestacked=False for unstacked observations."
        )

    return [frame[..., index] for index in range(frame.shape[-1])]


def _to_viewable(frame: np.ndarray) -> np.ndarray:
    """Bring an observation back to uint8, so a preprocessed one is still watchable.

    Float frames are read as spanning [0, 1], which is what normalize produces.
    """
    if frame.dtype == np.uint8:
        return frame

    return ObservationPreprocessor.to_uint8(np.asarray(frame))


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Return a writable copy of a stored frame in the channel order OpenCV expects."""
    if frame.ndim == 3 and frame.shape[2] == 3:
        # Observations are stored RGB; this undoes the conversion done on capture.
        frame = frame[:, :, ::-1]

    # A copy rather than a view, because stored frames are memory-mapped read-only and
    # the overlay is drawn into the frame in place.
    return np.array(frame, order="C")


def _tile(images: Sequence[np.ndarray]) -> np.ndarray:
    """Join equally sized images into one row, separated by a thin grey line."""
    images = list(images)

    if len(images) == 1:
        return images[0]

    height, _, channels = images[0].shape
    separator = np.full(
        (height, TILE_SEPARATOR, channels), TILE_SEPARATOR_VALUE, dtype=images[0].dtype
    )

    row = [images[0]]

    for image in images[1:]:
        row.append(separator)
        row.append(image)

    return np.ascontiguousarray(np.concatenate(row, axis=1))


def _draw_overlay(
    cv2,
    frame: np.ndarray,
    index: int,
    total: int,
    timestamp: Optional[float],
    action: Any,
    action_names: Optional[Sequence[str]],
) -> None:
    """Draw the step and action of a frame onto it, in place."""
    elapsed = f"{float(timestamp):.2f}s" if _is_number(timestamp) else "n/a"
    lines = [
        f"step {index + 1}/{total}  t={elapsed}",
        f"action {action}: {_action_label(action, action_names)}",
    ]

    font_scale = _font_scale(frame)
    line_height = int(30 * font_scale)

    for number, text in enumerate(lines, start=1):
        origin = (int(10 * font_scale), line_height * number)
        _put_text(cv2, frame, text, origin, font_scale)


def _draw_stack_labels(cv2, frame: np.ndarray, count: int) -> None:
    """Caption each frame of a tiled stack with its age, in place.

    The tiles are equally wide, so their positions follow from the count alone and stay
    right whether or not the image was scaled.
    """
    if count < 2:
        return

    font_scale = _font_scale(frame)
    height, width = frame.shape[0], frame.shape[1]
    tile_width = width / count
    baseline = height - int(8 * font_scale)

    for position in range(count):
        age = count - 1 - position
        text = "t" if age == 0 else f"t-{age}"
        origin = (int(position * tile_width + 5 * font_scale), baseline)

        _put_text(cv2, frame, text, origin, font_scale)


def _font_scale(frame: np.ndarray) -> float:
    """Size the overlay text to the image it is drawn on."""
    return max(0.4, min(1.2, frame.shape[0] / 720))


def _put_text(cv2, frame: np.ndarray, text: str, origin, font_scale: float) -> None:
    """Write light text with a dark outline, so it stays readable over any frame."""
    strokes = (
        ((0, 0, 0), max(2, int(4 * font_scale))),
        ((255, 255, 255), max(1, int(2 * font_scale))),
    )

    for color, thickness in strokes:
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _action_label(action: Any, action_names: Optional[Sequence[str]]) -> str:
    """Name an action index, using the key list that produced it."""
    if not _is_number(action):
        return str(action)

    index = int(action)

    # KeyboardActionCapture maps keys[i] to i + 1, leaving 0 for "nothing was done".
    if index == 0:
        return "noop"

    if action_names and 1 <= index <= len(action_names):
        return str(action_names[index - 1])

    return str(action)


def _wait(cv2, seconds: float) -> bool:
    """Show the current frame for a while, returning False if playback should stop."""
    key = cv2.waitKey(max(1, int(round(seconds * 1000)))) & 0xFF

    if key in (ord("q"), 27):
        return False

    if key == ord(" "):
        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord(" "):
                break

            if key in (ord("q"), 27):
                return False

    return True


if __name__ == "__main__":
    # Example usage: record a demonstration, store it, and replay it.
    from copy_cat.action_capture.keyboard_action_capture import KeyboardActionCapture
    from copy_cat.observation_capture.screen_observation_capture import (
        ScreenObservationCapture,
    )
    from copy_cat.recorder.demonstration_recorder import DemonstrationRecorder

    store = TrajectoryStore("./recordings")

    with ScreenObservationCapture(monitor=1) as observation_capture, KeyboardActionCapture(
        keys=["w", "a", "s", "d", "space"]
    ) as action_capture:
        recorder = DemonstrationRecorder(observation_capture, action_capture, toggle_key="9")
        trajectory = recorder.record(fps=10)

    episode = store.save(trajectory, fps=10)

    print("Saved episode:", episode)
    print("Stored episodes:", store.list())

    store.play(episode, scale=0.5)
