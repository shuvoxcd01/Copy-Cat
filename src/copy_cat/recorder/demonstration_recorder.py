import threading
import time

from pynput import keyboard

from copy_cat.action_capture.action_capture import ActionCapture
from copy_cat.observation_capture.observation_capture import ObservationCapture
from copy_cat.utils.key_names import key_lookups, normalize_key_name
from copy_cat.utils.trajectory import Trajectory


class DemonstrationRecorder:
    """Records demonstrations by pairing screen observations with user actions.

    A global toggle key starts and stops a session, so recording can be controlled
    without leaving the application being demonstrated. The first press starts the
    session and the second one ends it, at which point record() returns the
    Trajectory it built.

    Each step grabs an observation, waits one interval, then polls the action taken
    during that interval, so the recorded action is the response to the observation
    stored with it rather than to the previous one.

    Most intervals of a demonstration are idle, so the no-op action usually dominates
    a trajectory. Setting strip_noop discards those steps along with their
    observations, which leaves the trajectory unevenly sampled in time and removes
    "do nothing" from the label set, so a policy trained on it never learns to stay
    still. Timestamps keep their true offsets, so the gaps stay visible.

    The capture objects are not owned by the recorder and are never closed by it,
    so a single recorder can produce several trajectories.
    """

    def __init__(
        self,
        observation_capture: ObservationCapture,
        action_capture: ActionCapture,
        toggle_key: str = "f9",
        strip_noop: bool = False,
        verbose: bool = True,
    ):
        self.toggle_key = normalize_key_name(toggle_key)
        self._toggle_special, self._toggle_chars = key_lookups(self.toggle_key)

        tracked = getattr(action_capture, "keys", ())

        if self.toggle_key in tracked:
            raise ValueError(
                f"Toggle key {self.toggle_key!r} is also a tracked action key, "
                "so pressing it would be recorded as an action."
            )

        self.observation_capture = observation_capture
        self.action_capture = action_capture
        self.strip_noop = strip_noop
        self.verbose = verbose
        self.last_trajectory = None

        self._toggle = threading.Event()
        self._toggle_held = False
        self._listener = None

    def record(self, fps: float = 10.0) -> Trajectory:
        """Record a demonstration between two presses of the toggle key."""
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")

        period = 1.0 / fps
        trajectory = Trajectory()

        # Exposed before the loop so the steps survive an interrupted session.
        self.last_trajectory = trajectory

        self._toggle.clear()
        self._toggle_held = False
        self._start_toggle_listener()

        try:
            self._log(f"Press {self.toggle_key} to start recording...")
            self._toggle.wait()
            self._toggle.clear()

            # Discard anything pressed before the session, including the toggle itself.
            self.action_capture.capture_action()

            self._log(f"Recording at {fps:g} Hz. Press {self.toggle_key} again to stop.")

            # Resolved once, and defensively, so a capture predating the ABC still works.
            noop = getattr(self.action_capture, "noop_action", 0)

            started = time.perf_counter()
            next_step = started
            polls = 0
            dropped = 0

            while not self._toggle.is_set():
                observation = self.observation_capture.capture_observation()

                next_step += period
                remaining = next_step - time.perf_counter()

                if remaining > 0:
                    time.sleep(remaining)
                else:
                    # Fell behind the requested rate; resync instead of catching up.
                    next_step = time.perf_counter()

                action = self.action_capture.capture_action()
                polls += 1

                print(f"Recorded step: action={action}, observation_shape={observation.shape}")

                if self.strip_noop and action == noop:
                    # Drop the frame with the action so the pair stays consistent.
                    dropped += 1
                    continue

                trajectory.record_step(
                    state=observation,
                    action=action,
                    reward=None,
                    timestamp=time.perf_counter() - started,
                )

            elapsed = time.perf_counter() - started
            steps = trajectory.get_trajectory_length()

            # Rate of polling, not of recording, so stripping does not look like slowness.
            achieved = polls / elapsed if elapsed > 0 else 0.0

            summary = f"Stopped: {steps} steps in {elapsed:.1f}s ({achieved:.1f} Hz achieved"

            if self.strip_noop:
                summary += f", {dropped} no-op steps dropped"

            self._log(summary + ").")
        finally:
            self._stop_toggle_listener()

        # The recorded indices are meaningless without the key list that produced them.
        trajectory.action_names = getattr(self.action_capture, "keys", None)

        return trajectory

    def _start_toggle_listener(self):
        """Start listening for the toggle key in the background."""
        if self._listener is not None:
            return

        # suppress=False so the toggle key still reaches the application in focus.
        self._listener = keyboard.Listener(
            on_press=self._on_toggle_press,
            on_release=self._on_toggle_release,
            suppress=False,
        )
        self._listener.start()
        self._listener.wait()

    def _stop_toggle_listener(self):
        """Stop the toggle listener and release its resources."""
        if self._listener is not None:
            self._listener.stop()
            self._listener.join()
            self._listener = None

    def _on_toggle_press(self, key):
        """Signal a toggle press, ignoring the repeats the OS sends while it is held."""
        if not self._is_toggle(key) or self._toggle_held:
            return

        self._toggle_held = True
        self._toggle.set()

    def _on_toggle_release(self, key):
        """Allow the next press of the toggle key to register."""
        if self._is_toggle(key):
            self._toggle_held = False

    def _is_toggle(self, key) -> bool:
        """Return whether a key event refers to the toggle key."""
        if isinstance(key, keyboard.Key):
            return key == self._toggle_special

        char = getattr(key, "char", None)

        return char is not None and char.lower() in self._toggle_chars

    def _log(self, message: str):
        """Report session progress when running verbosely."""
        if self.verbose:
            print(message)


if __name__ == "__main__":
    # Example usage
    from copy_cat.action_capture.keyboard_action_capture import KeyboardActionCapture
    from copy_cat.observation_capture.screen_observation_capture import (
        ScreenObservationCapture,
    )

    with ScreenObservationCapture(monitor=1) as observation_capture, KeyboardActionCapture(
        keys=["w", "a", "s", "d", "space"]
    ) as action_capture:
        recorder = DemonstrationRecorder(observation_capture, action_capture, toggle_key="9")
        trajectory = recorder.record(fps=10)

        print("Recorded steps:", trajectory.get_trajectory_length())
        print("Action names:", trajectory.action_names)

        if trajectory.get_trajectory_length() > 0:
            state, action, reward, info = trajectory.get_step_with_info(0)
            print("First step:", state.shape, action, reward, info)

        
