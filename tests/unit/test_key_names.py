import pytest
from pynput import keyboard

from copy_cat.utils.key_names import key_lookups, normalize_key_name


class TestNormalizeKeyName:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("W", "w"),
            ("w", "w"),
            ("Space", "space"),
            ("  F9 ", "f9"),
            ("SHIFT_R", "shift_r"),
        ],
    )
    def test_lowercases_and_strips(self, name, expected):
        assert normalize_key_name(name) == expected

    def test_single_space_character_is_kept(self):
        # Stripping would turn the literal " " key into an empty name.
        assert normalize_key_name(" ") == " "

    @pytest.mark.parametrize("name", ["", None, 5, b"w"])
    def test_rejects_non_strings_and_empty(self, name):
        with pytest.raises(ValueError, match="non-empty strings"):
            normalize_key_name(name)


class TestKeyLookups:
    def test_single_character_has_no_special_key(self):
        assert key_lookups("A") == (None, ("a",))

    def test_special_key_without_a_character(self):
        special, chars = key_lookups("F9")

        assert special is keyboard.Key.f9
        assert chars == ()

    def test_special_key_that_also_arrives_as_a_character(self):
        special, chars = key_lookups("space")

        assert special is keyboard.Key.space
        expected = keyboard.Key.space.value.char
        assert chars == (() if expected is None else (expected,))

    def test_left_and_right_modifiers_are_distinct(self):
        assert key_lookups("shift")[0] is keyboard.Key.shift
        assert key_lookups("shift_r")[0] is keyboard.Key.shift_r

    def test_unknown_name_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match=r"Unknown key name 'spacebar'.*space"):
            key_lookups("spacebar")

    def test_invalid_name_is_rejected(self):
        with pytest.raises(ValueError):
            key_lookups("")
