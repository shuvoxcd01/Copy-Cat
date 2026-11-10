"""Shared test doubles.

The real capture classes talk to the OS (pynput hooks the keyboard, mss grabs the
screen), which needs permissions and a display. These fakes stand in for them so
the tests are deterministic and run headless.
"""

import numpy as np
from pynput import keyboard


class FakeListener:
    """Stands in for pynput.keyboard.Listener without hooking the real keyboard.

    Tests drive it by calling press()/release(), which invoke the callbacks the
    code under test registered, exactly as the listener thread would.
    """

    instances = []
    # Optional hook run when start() is called, for tests that need a key pressed
    # as soon as the code under test begins listening.
    on_start = None

    def __init__(self, on_press=None, on_release=None, suppress=False):
        self.on_press = on_press
        self.on_release = on_release
        self.suppress = suppress
        self.started = False
        self.waited = False
        self.stopped = False
        self.joined = False
        FakeListener.instances.append(self)

    def start(self):
        self.started = True

        if FakeListener.on_start is not None:
            FakeListener.on_start(self)

    def wait(self):
        self.waited = True

    def stop(self):
        self.stopped = True

    def join(self):
        self.joined = True

    def press(self, key):
        self.on_press(key)

    def release(self, key):
        self.on_release(key)

    def tap(self, key):
        self.press(key)
        self.release(key)


class FakeController:
    """Stands in for pynput.keyboard.Controller without posting real keystrokes.

    Every press and release is logged in order as ("press", key) or ("release", key).
    """

    instances = []

    def __init__(self):
        self.events = []
        FakeController.instances.append(self)

    def press(self, key):
        self.events.append(("press", key))

    def release(self, key):
        self.events.append(("release", key))


def char(value):
    """Return a pynput event for a printable key."""
    return keyboard.KeyCode.from_char(value)


class FakeClock:
    """A controllable replacement for the time module's perf_counter and sleep."""

    def __init__(self, start=100.0):
        self.now = start
        self.sleeps = []

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds):
        self.now += seconds


def frame(value=0, shape=(2, 3, 3)):
    """Return a small constant uint8 frame."""
    return np.full(shape, value, dtype=np.uint8)
