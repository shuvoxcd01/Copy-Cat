import numpy as np

from copy_cat.action_capture.keyboard_action_capture import KeyboardActionCapture
from copy_cat.observation_capture.screen_observation_capture import (
    ScreenObservationCapture,
)
from copy_cat.recorder.demonstration_recorder import DemonstrationRecorder
from copy_cat.storage.trajectory_store import TrajectoryStore
from copy_cat.preprocessing.action_preprocessor import (
    ActionPreprocessor,
    num_actions_from_names,
)
from copy_cat.preprocessing.observation_preprocessor import ObservationPreprocessor
from functools import partial

store = TrajectoryStore("./recordings/TRexRunner")

print(store.list())
selected_episode = store.list()[1]

# trajectory = store.load(selected_episode)

# The episodes dataset_creator.py writes are 4-deep grayscale stacks, so the stack is
# drawn frame by frame; scale up, because 84x84 tiles are tiny on a screen.
store.play(name=selected_episode, scale=2.0, is_framestacked=True)

