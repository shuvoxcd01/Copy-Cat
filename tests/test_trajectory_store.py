import json
import sys

import numpy as np
import pytest

from copy_cat.storage import trajectory_store
from copy_cat.storage.trajectory_store import (
    DEFAULT_FPS,
    META_FILE,
    OBSERVATIONS_FILE,
    STEPS_FILE,
    TrajectoryStore,
    frame_delays,
)
from copy_cat.utils.trajectory import Trajectory


def make_trajectory(steps=3, shape=(4, 4, 3), action_names=("w", "a", "s")):
    """Build a small trajectory that looks like a recorded demonstration."""
    rng = np.random.default_rng(0)
    trajectory = Trajectory()

    for step in range(steps):
        trajectory.record_step(
            state=rng.integers(0, 255, size=shape, dtype=np.uint8),
            action=step % 3,
            reward=None,
            timestamp=round(step * 0.1, 3),
        )

    trajectory.action_names = action_names

    return trajectory


class TestSaveLoad:
    def test_round_trip_preserves_everything(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        original = make_trajectory(steps=5)

        store.save(original, name="run_01")
        loaded = store.load("run_01")

        assert loaded.get_trajectory_length() == 5
        assert loaded.action_names == ("w", "a", "s")

        for step in range(5):
            assert np.array_equal(loaded.get_state(step), original.get_state(step))
            assert loaded.state_actions[step][1] == original.state_actions[step][1]
            assert loaded.rewards[step] == original.rewards[step]
            assert loaded.additional_info[step] == original.additional_info[step]

    def test_rewards_survive(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = make_trajectory(steps=2)
        trajectory.rewards = [1.5, -2.0]

        store.save(trajectory, name="rewarded")

        assert store.load("rewarded").rewards == [1.5, -2.0]

    def test_numpy_values_in_actions_and_info(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = Trajectory()
        trajectory.record_step(
            state=np.zeros((2, 2, 3), dtype=np.uint8),
            action=np.int64(2),
            reward=np.float32(0.5),
            timestamp=np.float64(0.25),
            extras=np.array([1, 2]),
        )

        store.save(trajectory, name="numpy_values")
        loaded = store.load("numpy_values")

        assert loaded.state_actions[0][1] == 2
        assert loaded.rewards[0] == pytest.approx(0.5)
        assert loaded.additional_info[0]["timestamp"] == pytest.approx(0.25)
        assert loaded.additional_info[0]["extras"] == [1, 2]

    def test_mmap_by_default_and_optional(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(), name="run_01")

        assert isinstance(store.load("run_01").get_state(0), np.memmap)

        eager = store.load("run_01", mmap=False).get_state(0)

        assert isinstance(eager, np.ndarray)
        assert not isinstance(eager, np.memmap)

    def test_layout_on_disk(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=3), name="run_01", fps=10.0, metadata={"game": "x"})

        episode = tmp_path / "run_01"
        meta = json.loads((episode / META_FILE).read_text())
        lines = (episode / STEPS_FILE).read_text().strip().splitlines()

        assert (episode / OBSERVATIONS_FILE).is_file()
        assert meta["num_steps"] == 3
        assert meta["fps"] == 10.0
        assert meta["obs_shape"] == [4, 4, 3]
        assert meta["obs_dtype"] == "uint8"
        assert meta["action_names"] == ["w", "a", "s"]
        assert meta["metadata"] == {"game": "x"}
        assert len(lines) == 3
        assert json.loads(lines[0]) == {
            "step": 0,
            "action": 0,
            "reward": None,
            "info": {"timestamp": 0.0},
        }

    def test_observations_are_a_plain_npy_stack(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = make_trajectory(steps=3)
        store.save(trajectory, name="run_01")

        frames = np.load(tmp_path / "run_01" / OBSERVATIONS_FILE)

        assert frames.shape == (3, 4, 4, 3)
        assert frames.dtype == np.uint8
        assert np.array_equal(frames[1], trajectory.get_state(1))

    def test_fps_inferred_from_timestamps(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=5), name="run_01")

        assert store.info("run_01")["fps"] == pytest.approx(10.0)

    def test_empty_trajectory(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(Trajectory(), name="empty")

        meta = store.info("empty")
        loaded = store.load("empty")

        assert meta["num_steps"] == 0
        assert meta["obs_shape"] is None
        assert meta["fps"] is None
        assert not (tmp_path / "empty" / OBSERVATIONS_FILE).exists()
        assert loaded.get_trajectory_length() == 0
        assert loaded.action_names is None

    def test_trajectory_without_action_names(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = make_trajectory(steps=1, action_names=None)

        store.save(trajectory, name="run_01")

        assert store.info("run_01")["action_names"] is None
        assert store.load("run_01").action_names is None

    def test_missing_episode(self, tmp_path):
        store = TrajectoryStore(tmp_path)

        with pytest.raises(FileNotFoundError):
            store.load("nope")

    def test_inconsistent_episode_is_rejected(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=3), name="run_01")

        steps_file = tmp_path / "run_01" / STEPS_FILE
        kept = steps_file.read_text().strip().splitlines()[:2]
        steps_file.write_text("\n".join(kept) + "\n")

        with pytest.raises(ValueError, match="inconsistent"):
            store.load("run_01")


class TestNaming:
    def test_generated_names_do_not_collide(self, tmp_path):
        store = TrajectoryStore(tmp_path)

        first = store.save(make_trajectory(steps=1))
        second = store.save(make_trajectory(steps=1))

        assert first != second
        assert sorted(store.list()) == sorted([first, second])

    @pytest.mark.parametrize(
        "name", ["../escape", "nested/run", "", "   ", ".hidden", " run_01", "."]
    )
    def test_rejects_unsafe_names(self, tmp_path, name):
        store = TrajectoryStore(tmp_path)

        with pytest.raises(ValueError):
            store.save(make_trajectory(steps=1), name=name)

    def test_nothing_is_written_outside_the_root(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")

        with pytest.raises(ValueError):
            store.save(make_trajectory(steps=1), name="../escape")

        assert not (tmp_path / "escape").exists()


class TestManagement:
    def test_list_exists_delete(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=1), name="b_run")
        store.save(make_trajectory(steps=1), name="a_run")

        assert store.list() == ["a_run", "b_run"]
        assert store.exists("a_run")
        assert "a_run" in store
        assert len(store) == 2
        assert list(store) == ["a_run", "b_run"]

        store.delete("a_run")

        assert store.list() == ["b_run"]
        assert not store.exists("a_run")

    def test_list_ignores_unrelated_directories(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=1), name="run_01")
        (tmp_path / "not_an_episode").mkdir()
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "loose_file.txt").write_text("x")

        assert store.list() == ["run_01"]

    def test_overwrite_requires_the_flag(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=3), name="run_01")

        with pytest.raises(FileExistsError):
            store.save(make_trajectory(steps=1), name="run_01")

        assert store.load("run_01").get_trajectory_length() == 3

        store.save(make_trajectory(steps=1), name="run_01", overwrite=True)

        assert store.load("run_01").get_trajectory_length() == 1

    def test_failed_save_leaves_the_previous_episode_intact(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=3), name="run_01")

        broken = make_trajectory(steps=2)
        broken.state_actions[1] = (np.zeros((9, 9, 3), dtype=np.uint8), 0)

        with pytest.raises(ValueError):
            store.save(broken, name="run_01", overwrite=True)

        assert store.load("run_01").get_trajectory_length() == 3
        assert store.list() == ["run_01"]

    def test_root_is_created_or_required(self, tmp_path):
        TrajectoryStore(tmp_path / "made" / "here")

        assert (tmp_path / "made" / "here").is_dir()

        with pytest.raises(FileNotFoundError):
            TrajectoryStore(tmp_path / "absent", create=False)


class TestObservationValidation:
    def test_rejects_mismatched_shapes(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = make_trajectory(steps=2)
        trajectory.state_actions[1] = (np.zeros((8, 8, 3), dtype=np.uint8), 0)

        with pytest.raises(ValueError, match="step 1"):
            store.save(trajectory, name="ragged")

    def test_rejects_mismatched_dtypes(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = make_trajectory(steps=2)
        trajectory.state_actions[1] = (np.zeros((4, 4, 3), dtype=np.float32), 0)

        with pytest.raises(ValueError, match="step 1"):
            store.save(trajectory, name="mixed")

    def test_rejects_non_array_observations(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = make_trajectory(steps=2)
        trajectory.state_actions[1] = ("not a frame", 0)

        with pytest.raises(TypeError, match="step 1"):
            store.save(trajectory, name="bad")

    def test_rejects_unserialisable_step_metadata(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = Trajectory()
        trajectory.record_step(
            state=np.zeros((2, 2, 3), dtype=np.uint8), action=0, reward=None, when=object()
        )

        with pytest.raises(TypeError):
            store.save(trajectory, name="bad_info")

        assert store.list() == []


class FakeCv2:
    """Stands in for OpenCV so playback can be tested without a display."""

    WINDOW_NORMAL = 0
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16
    INTER_AREA = 3

    def __init__(self, keys=()):
        self.keys = list(keys)
        self.windows = []
        self.destroyed = []
        self.shown = []
        self.waits = []
        self.texts = []

    def namedWindow(self, name, flags=0):
        self.windows.append(name)

    def imshow(self, name, frame):
        self.shown.append(frame.copy())

    def waitKey(self, delay):
        self.waits.append(delay)

        return self.keys.pop(0) if self.keys else -1

    def destroyWindow(self, name):
        self.destroyed.append(name)

    def resize(self, frame, size, fx=1.0, fy=1.0, interpolation=0):
        return frame[:: int(1 / fy), :: int(1 / fx)]

    def putText(self, frame, text, origin, font, scale, color, thickness, line_type):
        self.texts.append(text)


class TestPlay:
    @pytest.fixture
    def store(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=3), name="run_01", fps=10.0)

        return store

    def use(self, monkeypatch, fake):
        monkeypatch.setattr(trajectory_store, "_import_cv2", lambda: fake)

        return fake

    def test_shows_every_frame_in_order(self, store, monkeypatch):
        fake = self.use(monkeypatch, FakeCv2())

        store.play("run_01")

        original = store.load("run_01")

        assert len(fake.shown) == 3
        assert fake.windows == ["copy-cat: run_01"]
        assert fake.destroyed == ["copy-cat: run_01"]

        for index, shown in enumerate(fake.shown):
            # Frames are stored RGB and handed to OpenCV as BGR.
            assert np.array_equal(shown[:, :, ::-1], np.asarray(original.get_state(index)))

    def test_paces_from_recorded_timestamps(self, store, monkeypatch):
        fake = self.use(monkeypatch, FakeCv2())

        store.play("run_01")

        # 100 ms steps, minus the negligible time spent drawing, plus the teardown poll.
        assert [wait for wait in fake.waits[:3]] == pytest.approx([100, 100, 100], abs=5)

    def test_fps_and_speed_override_the_pacing(self, store, monkeypatch):
        fake = self.use(monkeypatch, FakeCv2())

        store.play("run_01", fps=2.0, speed=2.0)

        assert fake.waits[:3] == pytest.approx([250, 250, 250], abs=5)

    def test_overlay_can_be_turned_off(self, store, monkeypatch):
        fake = self.use(monkeypatch, FakeCv2())

        store.play("run_01", overlay=True)
        drawn = list(fake.texts)

        fake.texts.clear()
        store.play("run_01", overlay=False)

        assert fake.texts == []
        # 0 is the no-op; keys[i] is action i + 1.
        assert "action 0: noop" in drawn
        assert "action 1: w" in drawn
        assert "action 2: a" in drawn
        assert any(text.startswith("step 1/3") for text in drawn)

    def test_quit_key_stops_early_and_closes_the_window(self, store, monkeypatch):
        fake = self.use(monkeypatch, FakeCv2(keys=[-1, ord("q")]))

        store.play("run_01")

        assert len(fake.shown) == 2
        assert fake.destroyed == ["copy-cat: run_01"]

    def test_space_pauses_until_resumed(self, store, monkeypatch):
        fake = self.use(monkeypatch, FakeCv2(keys=[ord(" "), -1, -1, ord(" ")]))

        store.play("run_01")

        assert len(fake.shown) == 3

    def test_loop_replays_until_quit(self, store, monkeypatch):
        fake = self.use(monkeypatch, FakeCv2(keys=[-1, -1, -1, -1, ord("q")]))

        store.play("run_01", loop=True)

        assert len(fake.shown) == 5

    def test_scale_shrinks_the_frames(self, store, monkeypatch):
        fake = self.use(monkeypatch, FakeCv2())

        store.play("run_01", scale=0.5)

        assert fake.shown[0].shape == (2, 2, 3)

    def test_empty_episode_opens_no_window(self, tmp_path, monkeypatch, capsys):
        store = TrajectoryStore(tmp_path)
        store.save(Trajectory(), name="empty")
        fake = self.use(monkeypatch, FakeCv2())

        store.play("empty")

        assert fake.windows == []
        assert "no steps" in capsys.readouterr().out

    def test_missing_episode(self, store, monkeypatch):
        self.use(monkeypatch, FakeCv2())

        with pytest.raises(FileNotFoundError):
            store.play("nope")

    def test_opencv_is_only_needed_for_playback(self, store, monkeypatch):
        # A None entry in sys.modules makes "import cv2" raise ImportError.
        monkeypatch.setitem(sys.modules, "cv2", None)

        with pytest.raises(ImportError, match=r"copy-cat\[replay\]"):
            store.play("run_01")

        # Saving and loading never touch OpenCV.
        assert store.load("run_01").get_trajectory_length() == 3


class TestFrameDelays:
    def test_follows_recorded_timestamps(self):
        delays = frame_delays([0.0, 0.1, 0.5, 0.6])

        assert delays == pytest.approx([0.1, 0.4, 0.1, 0.1])

    def test_explicit_fps_overrides_timestamps(self):
        assert frame_delays([0.0, 0.1, 0.5], fps=4.0) == pytest.approx([0.25] * 3)

    def test_speed_scales_the_delays(self):
        assert frame_delays([0.0, 0.2, 0.4], speed=2.0) == pytest.approx([0.1, 0.1, 0.1])

    def test_falls_back_when_timestamps_are_unusable(self):
        assert frame_delays([None, None]) == pytest.approx([1 / DEFAULT_FPS] * 2)
        assert frame_delays([0.0]) == pytest.approx([1 / DEFAULT_FPS])
        assert frame_delays([None, 0.2], fallback_fps=5.0) == pytest.approx([0.2, 0.2])

    def test_empty(self):
        assert frame_delays([]) == []

    def test_non_monotonic_timestamps_do_not_go_negative(self):
        assert frame_delays([0.0, 0.3, 0.1]) == pytest.approx([0.3, 0.0, 0.0])

    @pytest.mark.parametrize(
        "kwargs", [{"speed": 0}, {"speed": -1}, {"fps": 0}, {"fps": -5}, {"fallback_fps": 0}]
    )
    def test_rejects_non_positive_rates(self, kwargs):
        with pytest.raises(ValueError):
            frame_delays([0.0, 0.1], **kwargs)
