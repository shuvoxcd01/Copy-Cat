from typing import Optional, Sequence

from pynput import keyboard

from copy_cat.action_execution.action_executor import ActionExecutor
from copy_cat.utils.key_names import key_lookups, normalize_key_name


class KeyboardActionExecutor(ActionExecutor):
    """Presses the keys a policy asks for, inverting KeyboardActionCapture's mapping.

    Action indices mean what they meant while recording: 0 is the noop and
    action_names[i] is i + 1, so the key list stored with an episode is what turns a
    policy's output back into a keystroke.

    A key stays down while the policy keeps choosing it and is released as soon as the
    policy chooses another action, which reproduces a held key rather than turning it
    into one tap per step. Only one key is ever down, matching the single-key action
    space the capture side records.
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

        if action == self.noop_action:
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
