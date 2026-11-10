"""Fixtures shared by every test module; the doubles themselves live in fakes.py."""

import pytest
from pynput import keyboard

from fakes import FakeClock, FakeController, FakeListener


@pytest.fixture
def fake_listener(monkeypatch):
    """Replace pynput's Listener so no test hooks the real keyboard."""
    FakeListener.instances = []
    FakeListener.on_start = None
    monkeypatch.setattr(keyboard, "Listener", FakeListener)

    yield FakeListener

    FakeListener.on_start = None


@pytest.fixture
def fake_controller(monkeypatch):
    """Replace pynput's Controller so no test types into the real keyboard."""
    FakeController.instances = []
    monkeypatch.setattr(keyboard, "Controller", FakeController)

    return FakeController


@pytest.fixture
def fake_clock():
    return FakeClock()
