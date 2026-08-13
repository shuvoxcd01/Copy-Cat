import threading
from typing import Dict, Optional, Sequence

from pynput import keyboard

from copy_cat.action_capture.action_capture import ActionCapture
from copy_cat.utils.key_names import key_lookups, normalize_key_name


class KeyboardActionCapture(ActionCapture):
    """Concrete implementation of ActionCapture for keyboard actions.

    Key events arrive asynchronously on a background listener thread, while
    capture_action() is a synchronous poll. Events are accumulated between polls
    so a key tapped and released between two calls is still reported.

    Actions are discrete indices: 0 means no tracked key was active, and keys[i]
    maps to i + 1. Only one key can be reported per poll, so when several are
    active the most recently pressed one wins.

    Special keys are named after pynput's keyboard.Key members ("space", "up",
    "shift"); anything else is treated as a single character. Left and right
    modifiers are distinct, so "shift" does not match the right shift key.
    """

    def __init__(self, keys: Sequence[str]):
        if not keys:
            raise ValueError("At least one key must be tracked.")

        names = [self._normalize_name(key) for key in keys]

        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate keys are not allowed: {names}")

        self.keys = tuple(names)
        self._char_to_index: Dict[str, int] = {}
        self._key_to_index: Dict[keyboard.Key, int] = {}

        for index, name in enumerate(self.keys, start=1):
            self._register(name, index)

        self._lock = threading.Lock()
        self._held: Dict[int, int] = {}
        self._latched: Dict[int, int] = {}
        self._seq = 0
        self._listener = None

    def capture_action(self) -> int:
        """Capture the most recently activated tracked key as a discrete index."""
        if self._listener is None:
            self.start()

        with self._lock:
            active = dict(self._latched)

            for index, seq in self._held.items():
                if seq > active.get(index, -1):
                    active[index] = seq

            self._latched.clear()

        if not active:
            return 0

        return max(active, key=active.get)

    @property
    def num_actions(self) -> int:
        """Return the size of the discrete action space, including the no-op."""
        return len(self.keys) + 1

    def start(self):
        """Start listening for key events in the background."""
        if self._listener is not None:
            return

        # suppress=False so the demonstrated keystrokes still reach the target app.
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

        with self._lock:
            self._held.clear()
            self._latched.clear()

    def _on_press(self, key):
        """Record a key press on the listener thread."""
        index = self._resolve(key)

        if index is None:
            return

        with self._lock:
            # Ignore OS auto-repeat so a held key cannot outrank a newer press.
            if index in self._held:
                return

            self._seq += 1
            self._held[index] = self._seq
            self._latched[index] = self._seq

    def _on_release(self, key):
        """Record a key release, leaving the latch intact for the next poll."""
        index = self._resolve(key)

        if index is None:
            return

        with self._lock:
            self._held.pop(index, None)

    def _resolve(self, key) -> Optional[int]:
        """Return the action index for a key event, or None if it is untracked."""
        if isinstance(key, keyboard.Key):
            return self._key_to_index.get(key)

        char = getattr(key, "char", None)

        if char is None:
            return None

        return self._char_to_index.get(char.lower())

    def _register(self, name: str, index: int):
        """Map a key name onto its action index."""
        special, chars = key_lookups(name)

        if special is not None:
            self._key_to_index[special] = index

        for char in chars:
            self._char_to_index.setdefault(char, index)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a key name to its lookup form."""
        return normalize_key_name(name)

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager and release resources."""
        self.close()
