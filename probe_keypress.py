"""Probe: can this process make the dino jump, with no policy involved?

Everything evaluate.py does apart from the keystroke - screen capture, preprocessing, the
policy, the timing loop - is removed here, so a dino that does not jump narrows to one of
two things: macOS is refusing to post the event, or the event is going to a window other
than the game.

Run it from the same terminal you run evaluate.py from, because the macOS permission that
allows posting keystrokes is granted to that application rather than to Python.
"""

import time

from pynput import keyboard

PRESSES = 5
COUNTDOWN = 5
HOLD = 0.1


def is_trusted():
    """Report whether macOS lets this process post keystrokes, if it will say."""
    try:
        from ApplicationServices import AXIsProcessTrusted
    except ImportError:
        return None

    return bool(AXIsProcessTrusted())


trusted = is_trusted()

if trusted is None:
    print("Accessibility trust: unknown (ApplicationServices not importable).")
else:
    print(f"Accessibility trust: {trusted}.")

    if not trusted:
        print(
            "  macOS will silently drop every keystroke below. Add this terminal under\n"
            "  Privacy & Security > Accessibility, then quit and reopen it."
        )

controller = keyboard.Controller()

print(f"\nFocus the game window. Sending {PRESSES} spaces, one per second.")

for remaining in range(COUNTDOWN, 0, -1):
    print(f"  {remaining}...", flush=True)
    time.sleep(1)

for number in range(1, PRESSES + 1):
    controller.press(keyboard.Key.space)
    time.sleep(HOLD)
    controller.release(keyboard.Key.space)
    print(f"  sent space {number}/{PRESSES}", flush=True)
    time.sleep(1)

print("\nDone. If the dino never jumped, the keystrokes are not reaching Chrome.")
