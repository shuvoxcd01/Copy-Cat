import pytest
from pynput import keyboard

from copy_cat.utils.toggle_key import ToggleKey
from fakes import char


@pytest.fixture
def toggle(fake_listener):
    with ToggleKey("f9") as toggle:
        yield toggle


def listener(fake_listener):
    return fake_listener.instances[-1]


class TestLifecycle:
    def test_default_key_matches_the_recorder(self):
        assert ToggleKey().key == "f9"

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown key name"):
            ToggleKey("not_a_key")

    def test_start_launches_a_non_suppressing_listener(self, fake_listener, toggle):
        started = listener(fake_listener)

        assert started.started and started.waited
        assert started.suppress is False

    def test_start_twice_reuses_the_listener(self, fake_listener, toggle):
        toggle.start()

        assert len(fake_listener.instances) == 1

    def test_close_stops_and_joins_the_listener(self, fake_listener):
        with ToggleKey() as toggle:
            started = listener(fake_listener)

        assert started.stopped and started.joined
        assert toggle._listener is None

    def test_close_without_start_is_safe(self, fake_listener):
        ToggleKey().close()


class TestDetection:
    def test_press_sets_the_toggle(self, fake_listener, toggle):
        assert not toggle.is_set()

        listener(fake_listener).press(keyboard.Key.f9)

        assert toggle.is_set()

    def test_other_keys_are_ignored(self, fake_listener, toggle):
        listener(fake_listener).tap(keyboard.Key.space)
        listener(fake_listener).tap(char("a"))

        assert not toggle.is_set()

    def test_wait_returns_after_a_press_and_rearms(self, fake_listener, toggle):
        listener(fake_listener).tap(keyboard.Key.f9)

        toggle.wait()

        assert not toggle.is_set()

    def test_auto_repeat_is_ignored_until_release(self, fake_listener, toggle):
        keys = listener(fake_listener)
        keys.press(keyboard.Key.f9)
        toggle.wait()

        keys.press(keyboard.Key.f9)  # OS auto-repeat while still held
        assert not toggle.is_set()

        keys.release(keyboard.Key.f9)
        keys.press(keyboard.Key.f9)
        assert toggle.is_set()

    def test_character_toggle_matches_case_insensitively(self, fake_listener):
        with ToggleKey("9") as toggle:
            listener(fake_listener).press(char("9"))

            assert toggle.is_set()

        with ToggleKey("q") as toggle:
            listener(fake_listener).press(char("Q"))

            assert toggle.is_set()

    def test_close_clears_a_pending_press(self, fake_listener):
        toggle = ToggleKey()
        toggle.start()
        listener(fake_listener).press(keyboard.Key.f9)

        toggle.close()

        assert not toggle.is_set()
