from typing import Optional, Tuple

from pynput import keyboard


def normalize_key_name(name: str) -> str:
    """Normalize a key name to its lookup form."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"Key names must be non-empty strings, got {name!r}.")

    return name.lower() if len(name) == 1 else name.strip().lower()


def key_lookups(name: str) -> Tuple[Optional[keyboard.Key], Tuple[str, ...]]:
    """Return the pynput Key member and the characters that stand for a key name."""
    name = normalize_key_name(name)

    if len(name) == 1:
        return None, (name,)

    special = getattr(keyboard.Key, name, None)

    if special is None:
        valid = ", ".join(sorted(key.name for key in keyboard.Key))
        raise ValueError(f"Unknown key name {name!r}. Valid special keys: {valid}")

    # Some platforms deliver keys such as space as a plain character instead.
    chars = () if special.value.char is None else (special.value.char,)

    return special, chars
