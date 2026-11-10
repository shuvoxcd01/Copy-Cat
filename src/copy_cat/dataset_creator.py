import numpy as np

from copy_cat.action_capture.keyboard_action_capture import KeyboardActionCapture
from copy_cat.observation_capture.screen_observation_capture import (
    ScreenObservationCapture,
)
from copy_cat.recorder.demonstration_recorder import DemonstrationRecorder
from copy_cat.storage.trajectory_store import TrajectoryStore
from copy_cat.preprocessing.action_preprocessor import ActionPreprocessor, num_actions_from_names
from copy_cat.preprocessing.observation_preprocessor import ObservationPreprocessor
from functools import partial

store = TrajectoryStore("./recordings/TRexRunner/")
    
stages = [
    ("captured", lambda obs: obs),
    ("to_grayscale", ObservationPreprocessor.to_grayscale),
    ("downsample", partial(ObservationPreprocessor.downsample, factor=4, mode="mean")),
    # ("center_crop", partial(ObservationPreprocessor.center_crop, size=(84, 84))),
    ("resize", partial(ObservationPreprocessor.resize, size=(84, 84))),
    ("framestack", partial(ObservationPreprocessor.framestack, num_frames=4)),
    ("normalize", ObservationPreprocessor.normalize),
]

pipeline = ObservationPreprocessor.compose(*(stage for _, stage in stages))

with ScreenObservationCapture(monitor=1) as observation_capture, KeyboardActionCapture(
    keys=["space"]
) as action_capture:
    recorder = DemonstrationRecorder(observation_capture, action_capture, toggle_key="9")
    trajectory = recorder.record(fps=10)
    num_classes = num_actions_from_names(trajectory.action_names)


    state_actions = trajectory.state_actions
    observations = [step[0] for step in state_actions]
    actions = [step[1] for step in state_actions]

    observations_stacked = np.stack(observations, axis=0)
    processed_observations = pipeline(observations_stacked)

    preprocessed_actions = ActionPreprocessor.to_onehot(actions, num_classes=num_classes)

    trajectory.clear()

    for obs, act in zip(processed_observations, preprocessed_actions):
        trajectory.record_step(obs, act)


episode = store.save(trajectory, fps=10)

print("Saved episode:", episode)
print("Stored episodes:", store.list())

# store.play(episode, scale=0.5)