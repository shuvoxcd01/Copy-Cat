import pytest
from pynput import keyboard

from copy_cat.action_execution.action_executor import ActionExecutor
from copy_cat.action_execution.keyboard_action_executor import KeyboardActionExecutor


@pytest.fixture
def executor(fake_controller):
    return KeyboardActionExecutor(["space", "a"])


def events(executor):
    return executor._controller.events


class TestConstruction:
    def test_is_an_action_executor(self, executor):
        assert isinstance(executor, ActionExecutor)

    def test_action_space_includes_the_noop(self, executor):
        assert executor.num_actions == 3
        assert executor.noop_action == 0

    def test_names_are_normalized(self, fake_controller):
        assert KeyboardActionExecutor(["SPACE", "A"]).action_names == ("space", "a")

    def test_duplicate_keys_are_rejected(self, fake_controller):
        with pytest.raises(ValueError, match="Duplicate"):
            KeyboardActionExecutor(["a", "A"])

    def test_unknown_key_name_is_rejected(self, fake_controller):
        with pytest.raises(ValueError, match="Unknown key name"):
            KeyboardActionExecutor(["not_a_key"])


class TestApply:
    def test_noop_presses_nothing(self, executor):
        executor.apply(0)

        assert events(executor) == []

    def test_special_and_char_keys_map_to_action_indices(self, executor):
        executor.apply(1)
        executor.apply(2)

        assert events(executor) == [
            ("press", keyboard.Key.space),
            ("release", keyboard.Key.space),
            ("press", "a"),
        ]

    def test_repeating_an_action_holds_the_key(self, executor):
        for _ in range(3):
            executor.apply(1)

        assert events(executor) == [("press", keyboard.Key.space)]

    def test_noop_after_a_key_releases_it(self, executor):
        executor.apply(2)
        executor.apply(0)
        executor.apply(0)

        assert events(executor) == [("press", "a"), ("release", "a")]

    def test_numpy_style_ints_are_accepted(self, executor):
        executor.apply(1.0)

        assert events(executor) == [("press", keyboard.Key.space)]

    @pytest.mark.parametrize("action", [-1, 3])
    def test_out_of_range_action_raises(self, executor, action):
        with pytest.raises(ValueError, match="num_actions"):
            executor.apply(action)


class TestRelease:
    def test_release_with_nothing_held_does_nothing(self, executor):
        executor.release()

        assert events(executor) == []

    def test_leaving_the_context_releases_a_held_key(self, fake_controller):
        with KeyboardActionExecutor(["space"]) as executor:
            executor.apply(1)

        assert events(executor) == [
            ("press", keyboard.Key.space),
            ("release", keyboard.Key.space),
        ]

    def test_leaving_the_context_on_error_still_releases(self, fake_controller):
        with pytest.raises(RuntimeError):
            with KeyboardActionExecutor(["space"]) as executor:
                executor.apply(1)
                raise RuntimeError("boom")

        assert events(executor)[-1] == ("release", keyboard.Key.space)
