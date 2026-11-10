"""Run a trained policy live: capture the screen, ask the policy, press the key.

This is the other half of record_demonstrations.py. Recording pairs a screen frame with
the key a human pressed; this script replaces the human, feeding each captured frame
through the same preprocessing and pressing whatever the policy asks for.

Both scripts take their pipeline from config.build_pipeline(), because the policy only
ever saw observations of that exact shape and scale. The one deliberate difference is the
frame stack: recording has the whole episode in hand and stacks it at once, while here
frames arrive one at a time, so the streaming pipeline carries the history instead.

Keys are held rather than tapped (see KeyboardActionExecutor), and everything is released
when the session ends, including on Ctrl-C.

Run it with the game already open:

    python examples/trex_runner/run_policy.py

then press the toggle key to hand control to the policy, and press it again to take
control back.
"""

import time
import warnings
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

import config
from copy_cat.action_execution.keyboard_action_executor import KeyboardActionExecutor
from copy_cat.observation_capture.screen_observation_capture import (
    ScreenObservationCapture,
)
from copy_cat.policies.base_policy import BasePolicy
from copy_cat.preprocessing.action_preprocessor import (
    NOOP_ACTION,
    ActionPreprocessor,
    num_actions_from_names,
)
from copy_cat.preprocessing.observation_preprocessor import ObservationPreprocessor
from copy_cat.storage.trajectory_store import TrajectoryStore
from copy_cat.utils.toggle_key import ToggleKey

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

# Taking the policy's best action rather than sampling from it. A demonstration is mostly
# noop, so a behaviour-cloned policy is confident about doing nothing and a greedy action
# can sit still forever; set this to False to sample and keep the rare key presses the
# demonstrator made.
DETERMINISTIC = True


def load_policy(path: Union[str, Path] = config.POLICY_PATH) -> BasePolicy:
    """Load a trained policy and put it in evaluation mode."""
    policy = BasePolicy.load(str(path))

    if isinstance(policy, torch.nn.Module):
        policy.eval()

    return policy


def load_recording_settings(
    root: Union[str, Path] = config.RECORDINGS_ROOT,
) -> Tuple[Tuple[str, ...], float]:
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
            f"No episodes in {str(root)!r} to read the key list from; record one with "
            "record_demonstrations.py first."
        )

    meta = store.info(episodes[-1])
    action_names = meta.get("action_names")

    if not action_names:
        raise ValueError(
            f"Episode {episodes[-1]!r} carries no action_names, so its action indices "
            "cannot be turned back into keystrokes."
        )

    return tuple(action_names), float(meta.get("fps") or config.FPS)


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
            f"trained on {tuple(expected)}; the recordings were made with different "
            "preprocessing than config.build_pipeline() now applies."
        )


def run_policy(
    policy_path: Union[str, Path] = config.POLICY_PATH,
    recordings_root: Union[str, Path] = config.RECORDINGS_ROOT,
    monitor: int = config.MONITOR,
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
    pipeline = config.build_pipeline(streaming=True)
    period = 1.0 / fps

    with ScreenObservationCapture(monitor=monitor) as observation_capture, (
        KeyboardActionExecutor(action_names)
    ) as executor, ToggleKey(config.TOGGLE_KEY) as toggle:
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
    run_policy()
