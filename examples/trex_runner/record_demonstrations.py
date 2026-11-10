"""Record a TRex Runner demonstration and save it, already preprocessed, as an episode.

Open the game, run

    python examples/trex_runner/record_demonstrations.py

then press the toggle key to start recording and press it again to stop. Every frame is
passed through config.build_pipeline() and every action is one-hot encoded before the
episode is saved, so train_policy.py can use the episodes as they are.
"""

import numpy as np

import config
from copy_cat.action_capture.keyboard_action_capture import KeyboardActionCapture
from copy_cat.observation_capture.screen_observation_capture import (
    ScreenObservationCapture,
)
from copy_cat.preprocessing.action_preprocessor import (
    ActionPreprocessor,
    num_actions_from_names,
)
from copy_cat.recorder.demonstration_recorder import DemonstrationRecorder
from copy_cat.storage.trajectory_store import TrajectoryStore


def main() -> None:
    store = TrajectoryStore(config.RECORDINGS_ROOT)
    pipeline = config.build_pipeline()

    with ScreenObservationCapture(monitor=config.MONITOR) as observation_capture, (
        KeyboardActionCapture(keys=list(config.ACTION_KEYS))
    ) as action_capture:
        recorder = DemonstrationRecorder(
            observation_capture, action_capture, toggle_key=config.TOGGLE_KEY
        )
        trajectory = recorder.record(fps=config.FPS)

    if not trajectory.state_actions:
        print("Nothing was recorded, so no episode was saved.")
        return

    num_classes = num_actions_from_names(trajectory.action_names)

    observations = np.stack([state for state, _ in trajectory.state_actions], axis=0)
    actions = [action for _, action in trajectory.state_actions]

    processed_observations = pipeline(observations)
    encoded_actions = ActionPreprocessor.to_onehot(actions, num_classes=num_classes)

    trajectory.clear()

    for observation, action in zip(processed_observations, encoded_actions):
        trajectory.record_step(observation, action)

    episode = store.save(trajectory, fps=config.FPS)

    print("Saved episode:", episode)
    print("Stored episodes:", store.list())


if __name__ == "__main__":
    main()
