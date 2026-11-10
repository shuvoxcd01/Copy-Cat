import pytest
from pynput import keyboard

from copy_cat.action_capture.keyboard_action_capture import KeyboardActionCapture
from fakes import char

KEYS = ["w", "a", "space", "shift"]
W, A, SPACE, SHIFT = 1, 2, 3, 4


@pytest.fixture
def capture(fake_listener):
    capture = KeyboardActionCapture(KEYS)
    capture.start()

    yield capture

    capture.close()


@pytest.fixture
def listener(capture, fake_listener):
    return fake_listener.instances[-1]


class TestConstruction:
    def test_normalizes_and_stores_keys(self):
        capture = KeyboardActionCapture(["W", " Space "])

        assert capture.keys == ("w", "space")
        assert capture.num_actions == 3
        assert capture.noop_action == 0

    def test_requires_at_least_one_key(self):
        with pytest.raises(ValueError, match="At least one key"):
            KeyboardActionCapture([])

    def test_rejects_duplicates_after_normalization(self):
        with pytest.raises(ValueError, match="Duplicate"):
            KeyboardActionCapture(["w", "W"])

    def test_rejects_unknown_special_keys(self):
        with pytest.raises(ValueError, match="Unknown key name"):
            KeyboardActionCapture(["spacebar"])

    def test_does_not_listen_until_needed(self, fake_listener):
        KeyboardActionCapture(KEYS)

        assert fake_listener.instances == []


class TestListenerLifecycle:
    def test_start_launches_a_non_suppressing_listener(self, capture, listener):
        assert listener.started and listener.waited
        assert listener.suppress is False

    def test_start_is_idempotent(self, capture, fake_listener):
        capture.start()

        assert len(fake_listener.instances) == 1

    def test_first_poll_starts_the_listener(self, fake_listener):
        capture = KeyboardActionCapture(KEYS)

        assert capture.capture_action() == 0
        assert len(fake_listener.instances) == 1

        capture.close()

    def test_close_stops_the_listener_and_forgets_state(self, capture, listener):
        listener.press(char("w"))

        capture.close()

        assert listener.stopped and listener.joined
        assert capture._listener is None
        assert capture._held == {} and capture._latched == {}

    def test_close_without_start_is_harmless(self, fake_listener):
        KeyboardActionCapture(KEYS).close()

    def test_context_manager_closes(self, fake_listener):
        with KeyboardActionCapture(KEYS) as capture:
            capture.start()

        assert fake_listener.instances[0].stopped
        assert capture._listener is None


class TestPolling:
    def test_no_input_is_noop(self, capture):
        assert capture.capture_action() == 0

    @pytest.mark.parametrize(
        "event, expected",
        [
            (char("w"), W),
            (char("W"), W),  # shifted characters still count
            (char("a"), A),
            (keyboard.Key.space, SPACE),
            (keyboard.Key.shift, SHIFT),
        ],
    )
    def test_maps_keys_to_one_based_indices(self, capture, listener, event, expected):
        listener.press(event)

        assert capture.capture_action() == expected

    def test_untracked_keys_are_ignored(self, capture, listener):
        listener.tap(char("x"))
        listener.tap(keyboard.Key.enter)
        listener.tap(keyboard.Key.shift_r)  # right shift is not "shift"
        listener.tap(keyboard.KeyCode(vk=0))  # a key with no character

        assert capture.capture_action() == 0

    def test_space_delivered_as_a_character_counts(self, capture, listener):
        if keyboard.Key.space.value.char is None:
            pytest.skip("this platform never reports space as a character")

        listener.press(char(keyboard.Key.space.value.char))

        assert capture.capture_action() == SPACE

    def test_quick_tap_between_polls_is_latched_once(self, capture, listener):
        listener.tap(char("w"))

        assert capture.capture_action() == W
        assert capture.capture_action() == 0

    def test_held_key_is_reported_on_every_poll(self, capture, listener):
        listener.press(char("a"))

        assert capture.capture_action() == A
        assert capture.capture_action() == A

        listener.release(char("a"))

        assert capture.capture_action() == 0

    def test_most_recent_press_wins(self, capture, listener):
        listener.press(char("w"))
        listener.press(char("a"))

        assert capture.capture_action() == A

    def test_released_newer_tap_still_beats_older_held_key(self, capture, listener):
        listener.press(char("w"))
        listener.tap(char("a"))

        assert capture.capture_action() == A
        assert capture.capture_action() == W

    def test_auto_repeat_does_not_refresh_a_held_key(self, capture, listener):
        listener.press(char("w"))
        listener.press(char("a"))
        listener.press(char("w"))  # OS auto-repeat of the still-held w

        assert capture.capture_action() == A

    def test_repressing_after_release_counts_as_new(self, capture, listener):
        listener.press(char("w"))
        listener.press(char("a"))
        listener.release(char("w"))
        listener.press(char("w"))

        assert capture.capture_action() == W

    def test_release_of_untracked_key_is_ignored(self, capture, listener):
        listener.press(char("w"))
        listener.release(char("x"))

        assert capture.capture_action() == W
