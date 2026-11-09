"""Run a trained policy live: capture the screen, ask the policy, press the key.

This is the other half of dataset_creator.py. Recording pairs a screen frame with the key
a human pressed; evaluation replaces the human, feeding each captured frame through the
same preprocessing and pressing whatever the policy asks for.

The pipeline here must match the one the episodes were built with, because the policy only
ever saw observations of that exact shape and scale. The one deliberate difference is the
frame stack: dataset_creator has the whole episode in hand and uses framestack, while here
frames arrive one at a time, so framestack_stream carries the history instead.

Keys are held rather than tapped - the key for an action goes down when the policy first
chooses it and only comes back up when the policy chooses something else - so a policy
that cloned a demonstrator holding a key reproduces the hold. Everything is released when
the session ends, including on Ctrl-C.

Run it with the game already open:

    python -m copy_cat.evaluate

then press the toggle key to hand control to the policy, and press it again to take
control back.
"""

import threading
import time
from functools import partial
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from pynput import keyboard

from copy_cat.observation_capture.screen_observation_capture import (
    ScreenObservationCapture,
)
from copy_cat.policies.base_policy import BasePolicy
from copy_cat.storage.trajectory_store import TrajectoryStore
from copy_cat.utils.action_preprocessor import (
    ActionPreprocessor,
    NOOP_ACTION,
    num_actions_from_names,
)
from copy_cat.utils.key_names import key_lookups, normalize_key_name
from copy_cat.utils.observation_preprocessor import ObservationPreprocessor

import warnings

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

POLICY_PATH = "./trained_policy_6.pt"

# The recordings the policy was trained on, read for the key list and the recording rate.
RECORDINGS_ROOT = "./recordings/TRexRunner"

MONITOR = 1

# Falls back to this only if the episode metadata does not carry an fps.
DEFAULT_FPS = 10.0

NUM_STACKED_FRAMES = 4

# Pressed once to hand control to the policy and once more to take it back. It must not be
# a key the policy can press.
TOGGLE_KEY = "9"

# Sampling from the policy rather than taking its best action. A demonstration is mostly
# noop, so a behaviour-cloned policy is confident about doing nothing and a greedy action
# can sit still forever; sampling keeps the rare key presses the demonstrator made.
DETERMINISTIC = True


def build_pipeline(num_frames: int = NUM_STACKED_FRAMES):
    """Build the live preprocessing pipeline, matching the one that made the dataset.

    The returned pipeline carries a reset(), which clears the frame stack between
    sessions so a new run does not start with the last run's frames.
    """
    framestack = ObservationPreprocessor.framestack_stream(num_frames=num_frames)

    stages = [
        ("captured", lambda obs: obs),
        ("to_grayscale", ObservationPreprocessor.to_grayscale),
        ("downsample", partial(ObservationPreprocessor.downsample, factor=4, mode="mean")),
        # ("center_crop", partial(ObservationPreprocessor.center_crop, size=(84, 84))),
        ("resize", partial(ObservationPreprocessor.resize, size=(84, 84))),
        # framestack_stream, not framestack: one frame arrives at a time out here.
        ("framestack", framestack),
        ("normalize", ObservationPreprocessor.normalize),
    ]

    pipeline = ObservationPreprocessor.compose(*(stage for _, stage in stages))
    pipeline.reset = framestack.reset

    return pipeline


class KeyboardActionExecutor:
    """Presses the keys a policy asks for, inverting KeyboardActionCapture's mapping.

    Action indices mean what they meant while recording: 0 is the noop and action_names[i]
    is i + 1, so the key list stored with an episode is what turns a policy's output back
    into a keystroke.

    A key stays down while the policy keeps choosing it and is released as soon as the
    policy chooses another action, which reproduces a held key rather than turning it into
    one tap per step. Only one key is ever down, matching the single-key action space the
    capture side records.
    """

    def __init__(self, action_names: Sequence[str]):
        names = [normalize_key_name(name) for name in action_names]

        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate keys are not allowed: {names}")

        self.action_names = tuple(names)

        # Index 0 is the noop, which presses nothing, so the keys start at 1.
        self._keys = [None] + [self._as_key(name) for name in self.action_names]
        self._controller = keyboard.Controller()
        self._pressed: Optional[int] = None

    @property
    def num_actions(self) -> int:
        """Return the size of the action space this executor covers, with the noop."""
        return len(self._keys)

    def apply(self, action: int) -> None:
        """Make the keyboard reflect an action, holding it until the action changes."""
        action = int(action)

        if not 0 <= action < self.num_actions:
            raise ValueError(
                f"action must be below num_actions {self.num_actions}, got {action}."
            )

        if action == self._pressed:
            return

        self.release()

        if action == NOOP_ACTION:
            return

        self._controller.press(self._keys[action])
        self._pressed = action

    def release(self) -> None:
        """Let go of whatever is currently held, leaving the keyboard as it was found."""
        if self._pressed is None:
            return

        self._controller.release(self._keys[self._pressed])
        self._pressed = None

    @staticmethod
    def _as_key(name: str):
        """Return what pynput's controller should press for a key name."""
        special, chars = key_lookups(name)

        if special is not None:
            return special

        return chars[0]

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, releasing any key still held."""
        self.release()


class ToggleKey:
    """A global key that starts and stops a session, watched on a background thread.

    Control has to come from outside the window being played, since that window holds the
    keyboard focus, so this is the same global toggle DemonstrationRecorder uses to bound
    a recording.
    """

    def __init__(self, key: str = TOGGLE_KEY):
        self.key = normalize_key_name(key)
        self._special, self._chars = key_lookups(self.key)
        self._event = threading.Event()
        self._held = False
        self._listener = None

    def wait(self) -> None:
        """Block until the toggle key is pressed, then arm it for the next press."""
        self._event.wait()
        self._event.clear()

    def is_set(self) -> bool:
        """Say whether the toggle key has been pressed since the last wait()."""
        return self._event.is_set()

    def start(self):
        """Start listening for the toggle key in the background."""
        if self._listener is not None:
            return

        # suppress=False so the toggle key still reaches the application in focus.
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.start()
        self._listener.wait()

    def close(self):
        """Stop the listener and release its resources."""
        if self._listener is not None:
            self._listener.stop()
            self._listener.join()
            self._listener = None

        self._event.clear()
        self._held = False

    def _on_press(self, key):
        """Signal a press, ignoring the repeats the OS sends while the key is held."""
        if not self._matches(key) or self._held:
            return

        self._held = True
        self._event.set()

    def _on_release(self, key):
        """Allow the next press of the toggle key to register."""
        if self._matches(key):
            self._held = False

    def _matches(self, key) -> bool:
        """Return whether a key event refers to the toggle key."""
        if isinstance(key, keyboard.Key):
            return key == self._special

        char = getattr(key, "char", None)

        return char is not None and char.lower() in self._chars

    def __enter__(self):
        """Enter the context manager."""
        self.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager and release resources."""
        self.close()


def load_policy(path: str = POLICY_PATH) -> BasePolicy:
    """Load a trained policy and put it in evaluation mode."""
    policy = BasePolicy.load(path)

    if isinstance(policy, torch.nn.Module):
        policy.eval()

    return policy


def load_recording_settings(root: str = RECORDINGS_ROOT) -> Tuple[Tuple[str, ...], float]:
    """Return the key list and the rate the policy's training episodes were recorded at.

    The indices a policy emits only mean something alongside the keys that produced them,
    and a stack of frames only shows the motion the policy was trained on if the frames
    arrive at the rate they were recorded, so both come from the episodes themselves
    rather than from a constant here.
    """
    store = TrajectoryStore(root, create=False)
    episodes = store.list()

    if not episodes:
        raise ValueError(
            f"No episodes in {root!r} to read the key list from; record one with "
            "dataset_creator.py first."
        )

    meta = store.info(episodes[-1])
    action_names = meta.get("action_names")

    if not action_names:
        raise ValueError(
            f"Episode {episodes[-1]!r} carries no action_names, so its action indices "
            "cannot be turned back into keystrokes."
        )

    return tuple(action_names), float(meta.get("fps") or DEFAULT_FPS)


def select_action(
    policy: BasePolicy,
    observation: np.ndarray,
    deterministic: bool = DETERMINISTIC,
) -> Tuple[int, float]:
    """Ask the policy for an action, returning it with the probability it carried."""
    batch = ObservationPreprocessor.add_batch_dim(observation)
    tensor = torch.from_numpy(np.ascontiguousarray(batch))

    with torch.no_grad():
        logits = policy.forward(tensor)
        probabilities = torch.softmax(logits, dim=-1)

        if deterministic:
            action = int(torch.argmax(probabilities, dim=-1).item())
        else:
            action = int(torch.multinomial(probabilities, num_samples=1).item())

    return action, float(probabilities[0, action].item())


def check_observation(observation: np.ndarray, policy: BasePolicy) -> None:
    """Refuse an observation the policy was not trained on, before it reaches the game."""
    expected = getattr(policy, "observation_shape", None)

    if expected is not None and tuple(expected) != observation.shape:
        raise ValueError(
            f"The pipeline produces {observation.shape} observations but the policy was "
            f"trained on {tuple(expected)}; evaluate.py must preprocess exactly as "
            "dataset_creator.py did."
        )


def evaluate(
    policy_path: str = POLICY_PATH,
    recordings_root: str = RECORDINGS_ROOT,
    monitor: int = MONITOR,
    deterministic: bool = DETERMINISTIC,
    verbose: bool = True,
) -> None:
    """Hand the keyboard to a trained policy, between two presses of the toggle key."""
    policy = load_policy(policy_path)
    action_names, fps = load_recording_settings(recordings_root)
    num_actions = num_actions_from_names(action_names)

    trained_for = getattr(policy, "num_actions", num_actions)

    if trained_for != num_actions:
        raise ValueError(
            f"The policy has {trained_for} actions but the key list {list(action_names)} "
            f"describes {num_actions}; it was trained on a different recording."
        )

    labels = ActionPreprocessor.to_names(np.arange(num_actions), action_names).tolist()
    pipeline = build_pipeline()
    period = 1.0 / fps

    with ScreenObservationCapture(monitor=monitor) as observation_capture, (
        KeyboardActionExecutor(action_names)
    ) as executor, ToggleKey(TOGGLE_KEY) as toggle:
        if verbose:
            chosen = "greedily" if deterministic else "by sampling"
            print(f"Loaded {type(policy).__name__} over actions {labels}, acting {chosen}.")
            print(f"Focus the game, then press {toggle.key} to hand it over...")

        toggle.wait()

        # A new session starts with an empty frame stack, not the last session's frames.
        pipeline.reset()

        if verbose:
            print(f"Playing at {fps:g} Hz. Press {toggle.key} again to stop.")

        counts = np.zeros(num_actions, dtype=np.int64)
        started = time.perf_counter()
        next_step = started

        try:
            while not toggle.is_set():
                frame = observation_capture.capture_observation()
                observation = pipeline(frame)

                if not counts.any():
                    check_observation(observation, policy)

                action, probability = select_action(policy, observation, deterministic)
                executor.apply(action)
                counts[action] += 1

                if verbose and action != NOOP_ACTION:
                    step = int(counts.sum())
                    print(f"Step {step}: {labels[action]} (p={probability:.2f})")

                next_step += period
                remaining = next_step - time.perf_counter()

                if remaining > 0:
                    time.sleep(remaining)
                else:
                    # Fell behind the requested rate; resync instead of catching up.
                    next_step = time.perf_counter()
        finally:
            # Never leave a key down, however the session ended.
            executor.release()

        elapsed = time.perf_counter() - started
        steps = int(counts.sum())

        if verbose:
            achieved = steps / elapsed if elapsed > 0 else 0.0
            taken = ", ".join(
                f"{label} {count}" for label, count in zip(labels, counts.tolist())
            )
            print(
                f"Stopped: {steps} steps in {elapsed:.1f}s ({achieved:.1f} Hz achieved). "
                f"Actions taken: {taken}."
            )


if __name__ == "__main__":
    # Guarded, unlike the other scripts here, because importing this one would otherwise
    # start typing into whatever window happens to be in focus.
    evaluate()
