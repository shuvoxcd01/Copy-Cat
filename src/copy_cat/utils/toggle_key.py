import threading

from pynput import keyboard

from copy_cat.utils.key_names import key_lookups, normalize_key_name


class ToggleKey:
    """A global key that starts and stops a session, watched on a background thread.

    Control has to come from outside the window being played, since that window holds
    the keyboard focus, so this is the same kind of global toggle DemonstrationRecorder
    uses to bound a recording.
    """

    def __init__(self, key: str = "f9"):
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

    def start(self) -> None:
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

    def close(self) -> None:
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
