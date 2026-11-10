"""Settings shared by every TRex Runner script, so recording and playing cannot drift apart.

The policy only ever sees observations of the exact shape and scale the episodes were
built with, so the preprocessing pipeline is defined once here and used by both
record_demonstrations.py and run_policy.py.
"""

from functools import partial
from pathlib import Path

from copy_cat.preprocessing.observation_preprocessor import ObservationPreprocessor

EXAMPLE_DIR = Path(__file__).resolve().parent

RECORDINGS_ROOT = EXAMPLE_DIR / "recordings"
POLICY_PATH = EXAMPLE_DIR / "trained_policy.pt"

MONITOR = 1
FPS = 10.0

# The keys the demonstrator plays with. Action 0 is the noop and keys[i] is i + 1.
ACTION_KEYS = ("space",)

# Pressed once to start recording or hand control to the policy, and again to stop. It
# must not be one of ACTION_KEYS.
TOGGLE_KEY = "9"

NUM_STACKED_FRAMES = 4
DOWNSAMPLE_FACTOR = 4
FRAME_SIZE = (84, 84)

NUM_EPOCHS = 200


def build_pipeline(streaming: bool = False, num_frames: int = NUM_STACKED_FRAMES):
    """Build the observation preprocessing pipeline.

    With streaming=False the pipeline takes a whole episode stacked on axis 0, which is
    what record_demonstrations.py has once a recording ends. With streaming=True it takes
    one frame at a time, as run_policy.py captures them, and carries the frame history
    itself; the returned pipeline then has a reset() that clears that history between
    sessions.
    """
    if streaming:
        framestack = ObservationPreprocessor.framestack_stream(num_frames=num_frames)
    else:
        framestack = partial(ObservationPreprocessor.framestack, num_frames=num_frames)

    pipeline = ObservationPreprocessor.compose(
        ObservationPreprocessor.to_grayscale,
        partial(ObservationPreprocessor.downsample, factor=DOWNSAMPLE_FACTOR, mode="mean"),
        partial(ObservationPreprocessor.resize, size=FRAME_SIZE),
        framestack,
        ObservationPreprocessor.normalize,
    )

    if streaming:
        pipeline.reset = framestack.reset

    return pipeline
