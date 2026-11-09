# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Copy-Cat is a Python library (src layout, package `copy_cat`) for imitation learning: it records demonstrations by pairing screen captures with the user's keyboard actions, stores them on disk, and replays them.

## Commands

```bash
pip install -e '.[replay,test]'   # editable install; [replay] adds opencv-python (playback only), [test] adds pytest + pytest-cov

pytest                                            # run all tests
pytest --cov                                      # with branch coverage (config in pyproject.toml)
pytest tests/test_trajectory_store.py::TestPlay   # one test class
pytest tests/test_trajectory_store.py::TestPlay::test_loop_replays_until_quit  # one test
```

The package must be installed (or `src` put on `PYTHONPATH`) for `import copy_cat` to work in tests. No linter or formatter is configured.

Tests never touch real hardware. `tests/fakes.py` holds the doubles (`FakeListener` for pynput, `FakeClock` for `time`), and `tests/conftest.py` holds the fixtures that patch them in (`fake_listener`, `fake_clock`). mss and OpenCV are faked inside their own test modules. Known bugs are pinned as `xfail(strict=True)` tests, so a fix makes them XPASS and fail the run: remove the marker when you fix the bug.

Several modules have `if __name__ == "__main__":` demos (`python -m copy_cat.recorder.demonstration_recorder`, `python -m copy_cat.storage.trajectory_store`). They grab the real screen and keyboard, so on macOS they need Screen Recording and Input Monitoring/Accessibility permissions.

## Architecture

The data flow is **capture → record → store → replay**:

- **`observation_capture/`** and **`action_capture/`**: each defines an ABC (`ObservationCapture.capture_observation()`, `ActionCapture.capture_action()` plus `noop_action`) and one concrete implementation.
  - `ScreenObservationCapture` uses `mss` and returns **RGB** `uint8` arrays (it converts from mss's BGRA). It opens the mss handle lazily.
  - `KeyboardActionCapture` uses a background `pynput` listener; `capture_action()` is a synchronous *poll*. Presses are latched between polls so a quick tap isn't missed, OS auto-repeat is ignored, and if several keys are active the most recently pressed one wins. Actions are discrete ints: **0 = no-op, `keys[i]` → `i + 1`**. That mapping is relied on in several places (the recorder's `strip_noop`, `_action_label` in the store's overlay).
- **`utils/key_names.py`**: shared key-name normalization. Single characters are matched by char; longer names must be `pynput.keyboard.Key` member names (`"space"`, `"f9"`, `"shift"`). Left and right modifiers are distinct. Both the action capture and the recorder's toggle key go through it.
- **`recorder/DemonstrationRecorder`**: a global toggle key (default `f9`, and it may not also be a tracked action key) starts and stops a session. Each step grabs the observation, sleeps one period, *then* polls the action, so a stored action is the response to the observation stored with it. If the loop falls behind it resyncs rather than catching up. `strip_noop=True` drops no-op steps together with their frames, which leaves timestamps uneven on purpose. The recorder never closes the captures it is given. It sets `trajectory.action_names` (the key tuple) when it finishes, and `last_trajectory` exposes partial results if a session is interrupted.
- **`utils/trajectory.Trajectory`**: plain container of parallel lists `state_actions` (tuples of `(state, action)`), `rewards` and `additional_info` (per-step kwargs dicts; the recorder stores `timestamp` there). `action_names` is an ad-hoc attribute that isn't set in `__init__`, so read it with `getattr`. Watch out: `get_reward(t)` is **1-indexed** (`get_step(t)` calls `get_reward(t + 1)`), and `update_step` passes `kwargs=kwargs` through to `record_step`, which nests the dict.
- **`storage/TrajectoryStore`**: each episode is a directory `<root>/<name>/` containing `meta.json`, `steps.jsonl` (action, reward and info per line) and `observations.npy` (a single stacked array). The design is meant to handle multi-GB recordings:
  - Saves go through a `.tmp-<name>-<pid>` staging directory and are moved into place with `os.replace`, so a failed save leaves the old episode intact. Steps are written before frames so serialization errors show up early.
  - Frames are written one at a time into an `open_memmap` (never stacked in RAM), and `load()` memory-maps by default.
  - All observations in an episode must share one shape and dtype. Non-JSON values in actions/info are converted by `_jsonable` (numpy scalars and arrays are handled), and anything else raises.
  - Episode names are validated so they can't escape the root. A leading `.` is reserved for staging dirs.
  - `play()` imports OpenCV lazily via `_import_cv2()`, which is the seam the tests monkeypatch with `FakeCv2`. Playback timing comes from recorded timestamps (`frame_delays`) so gaps left by stripped no-ops are kept; the stored `fps` is only metadata and a fallback.
- **`utils/simple_replay_buffer.py`**: a standalone deque-based RL transition buffer. Nothing else uses it yet.
