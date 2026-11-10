import json
import os
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
from copy_cat.data.trajectory import Trajectory


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


class TestPlayFramestacked:
    """is_framestacked draws the frames of a stacked observation side by side."""

    def use(self, monkeypatch, fake):
        monkeypatch.setattr(trajectory_store, "_import_cv2", lambda: fake)

        return fake

    def tiles(self, shown, count, width):
        """Cut a shown image back into the tiles it was built from."""
        step = width + trajectory_store.TILE_SEPARATOR

        return [shown[:, index * step : index * step + width] for index in range(count)]

    def test_grayscale_stack_is_tiled_oldest_first(self, tmp_path, monkeypatch):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=2, shape=(4, 4, 3)), name="stacked")
        fake = self.use(monkeypatch, FakeCv2())

        store.play("stacked", is_framestacked=True, overlay=False)

        original = store.load("stacked")

        assert len(fake.shown) == 2

        for index, shown in enumerate(fake.shown):
            # Three 4x4 tiles, with a separator column between them.
            assert shown.shape == (4, 4 * 3 + 2 * trajectory_store.TILE_SEPARATOR, 3)

            stack = np.asarray(original.get_state(index))

            for position, tile in enumerate(self.tiles(shown, count=3, width=4)):
                # A grayscale tile is repeated across the three channels.
                assert np.array_equal(tile[:, :, 0], stack[:, :, position])
                assert np.array_equal(tile[:, :, 2], stack[:, :, position])

    def test_channelled_stack_splits_on_the_last_axis(self, tmp_path, monkeypatch):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=1, shape=(4, 4, 3, 2)), name="rgb_stack")
        fake = self.use(monkeypatch, FakeCv2())

        store.play("rgb_stack", is_framestacked=True, overlay=False)

        stack = np.asarray(store.load("rgb_stack").get_state(0))
        shown = fake.shown[0]

        assert shown.shape == (4, 4 * 2 + trajectory_store.TILE_SEPARATOR, 3)

        for position, tile in enumerate(self.tiles(shown, count=2, width=4)):
            # Frames are stored RGB and handed to OpenCV as BGR.
            assert np.array_equal(tile[:, :, ::-1], stack[:, :, :, position])

    def test_stacking_is_opt_in(self, tmp_path, monkeypatch):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=1, shape=(4, 4, 3)), name="stacked")
        fake = self.use(monkeypatch, FakeCv2())

        store.play("stacked")

        # Without the flag the same episode is one RGB frame, exactly as before.
        assert fake.shown[0].shape == (4, 4, 3)

    def test_tiles_are_labelled_with_their_age(self, tmp_path, monkeypatch):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=1, shape=(4, 4, 3)), name="stacked")
        fake = self.use(monkeypatch, FakeCv2())

        store.play("stacked", is_framestacked=True, overlay=True)
        drawn = list(fake.texts)

        fake.texts.clear()
        store.play("stacked", is_framestacked=True, overlay=False)

        # Every label is drawn twice, as light text over its own dark outline.
        labels = [text for text in drawn if text == "t" or text.startswith("t-")]

        assert fake.texts == []
        assert labels == ["t-2", "t-2", "t-1", "t-1", "t", "t"]

    def test_float_observations_are_shown_as_uint8(self, tmp_path, monkeypatch):
        store = TrajectoryStore(tmp_path)
        trajectory = Trajectory()
        state = np.zeros((4, 4, 2), dtype=np.float32)
        state[..., 1] = 1.0
        trajectory.record_step(state=state, action=0, reward=None, timestamp=0.0)
        store.save(trajectory, name="normalized")
        fake = self.use(monkeypatch, FakeCv2())

        store.play("normalized", is_framestacked=True, overlay=False)

        shown = fake.shown[0]
        oldest, newest = self.tiles(shown, count=2, width=4)

        assert shown.dtype == np.uint8
        assert np.array_equal(oldest, np.zeros_like(oldest))
        assert np.array_equal(newest, np.full_like(newest, 255))

    def test_rejects_an_observation_that_cannot_be_a_stack(self, tmp_path, monkeypatch):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=1, shape=(4, 4)), name="flat")
        self.use(monkeypatch, FakeCv2())

        with pytest.raises(ValueError, match=r"\(H, W, S\)"):
            store.play("flat", is_framestacked=True)


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


class TestDamagedEpisodes:
    def test_missing_observations_file_is_reported_on_load(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=2), name="run_01")
        (tmp_path / "run_01" / OBSERVATIONS_FILE).unlink()

        with pytest.raises(FileNotFoundError, match=OBSERVATIONS_FILE):
            store.load("run_01")

    def test_missing_observations_file_is_reported_on_play(self, tmp_path, monkeypatch):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=2), name="run_01")
        (tmp_path / "run_01" / OBSERVATIONS_FILE).unlink()
        fake = FakeCv2()
        monkeypatch.setattr(trajectory_store, "_import_cv2", lambda: fake)

        with pytest.raises(FileNotFoundError, match=OBSERVATIONS_FILE):
            store.play("run_01")

        assert fake.windows == []

    def test_corrupt_steps_name_the_line(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=2), name="run_01")
        steps_file = tmp_path / "run_01" / STEPS_FILE
        steps_file.write_text(steps_file.read_text() + "{not json\n")

        with pytest.raises(ValueError, match="corrupt at line 3"):
            store.load("run_01")

    def test_blank_lines_in_steps_are_skipped(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=2), name="run_01")
        steps_file = tmp_path / "run_01" / STEPS_FILE
        lines = steps_file.read_text().splitlines()
        steps_file.write_text(lines[0] + "\n\n   \n" + lines[1] + "\n\n")

        assert store.load("run_01").get_trajectory_length() == 2


class TestEdgeCases:
    def test_list_of_a_removed_root_is_empty(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        (tmp_path / "store").rmdir()

        assert store.list() == []
        assert len(store) == 0

    def test_container_protocol_and_repr(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=1), name="b")
        store.save(make_trajectory(steps=1), name="a")

        assert list(store) == ["a", "b"]
        assert "a" in store
        assert "missing" not in store
        assert 3 not in store
        assert repr(store) == f"TrajectoryStore(root={str(tmp_path)!r}, episodes=2)"

    def test_existing_root_is_accepted_without_create(self, tmp_path):
        assert TrajectoryStore(tmp_path, create=False).list() == []

    def test_rejects_a_non_array_first_step(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = Trajectory()
        trajectory.record_step(state=[[1, 2]], action=0, reward=None)

        with pytest.raises(TypeError, match="at step 0"):
            store.save(trajectory, name="bad")

    @pytest.mark.skipif(os.altsep is None, reason="only Windows has an alternative separator")
    def test_rejects_the_alternative_separator(self, tmp_path):
        with pytest.raises(ValueError, match="plain directory name"):
            TrajectoryStore(tmp_path).path_for(f"a{os.altsep}b")

    def test_rejects_a_non_array_after_the_first_step(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = make_trajectory(steps=2)
        trajectory.record_step(state=[[1, 2]], action=0, reward=None)

        with pytest.raises(TypeError, match="at step 2"):
            store.save(trajectory, name="bad")

        assert store.list() == []

    def test_numpy_bools_are_stored_as_bools(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        trajectory = Trajectory()
        trajectory.record_step(
            state=np.zeros((1, 1, 3), dtype=np.uint8), action=0, done=np.bool_(True)
        )

        store.save(trajectory, name="flags")

        assert store.load("flags").additional_info[0]["done"] is True

    def test_metadata_and_tuple_action_names_are_stored(self, tmp_path):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=1), name="run_01", metadata={"game": "demo", 1: (2, 3)})

        info = store.info("run_01")

        assert info["metadata"] == {"game": "demo", "1": [2, 3]}
        assert info["action_names"] == ["w", "a", "s"]

    @pytest.mark.parametrize(
        "timestamps",
        [
            [0.5, 0.5],  # no time passed
            [0.0, None, None],  # only one usable stamp
            [True, False],  # bools are not timestamps
        ],
    )
    def test_fps_is_not_inferred_from_unusable_timestamps(self, tmp_path, timestamps):
        store = TrajectoryStore(tmp_path)
        trajectory = Trajectory()

        for stamp in timestamps:
            trajectory.record_step(
                state=np.zeros((1, 1, 3), dtype=np.uint8), action=0, timestamp=stamp
            )

        store.save(trajectory, name="run_01")

        assert store.info("run_01")["fps"] is None

    def test_grayscale_frames_play_unchanged(self, tmp_path, monkeypatch):
        store = TrajectoryStore(tmp_path)
        trajectory = Trajectory()
        gray = np.arange(4, dtype=np.uint8).reshape(2, 2)
        trajectory.record_step(state=gray, action=0, timestamp=0.0)
        store.save(trajectory, name="gray")
        fake = FakeCv2()
        monkeypatch.setattr(trajectory_store, "_import_cv2", lambda: fake)

        store.play("gray", overlay=False)

        assert np.array_equal(fake.shown[0], gray)

    def test_space_pause_can_be_quit(self, tmp_path, monkeypatch):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=3), name="run_01")
        fake = FakeCv2(keys=[ord(" "), -1, 27])
        monkeypatch.setattr(trajectory_store, "_import_cv2", lambda: fake)

        store.play("run_01")

        assert len(fake.shown) == 1
        assert fake.destroyed == ["copy-cat: run_01"]

    def test_custom_window_name(self, tmp_path, monkeypatch):
        store = TrajectoryStore(tmp_path)
        store.save(make_trajectory(steps=1), name="run_01")
        fake = FakeCv2()
        monkeypatch.setattr(trajectory_store, "_import_cv2", lambda: fake)

        store.play("run_01", window="review")

        assert fake.windows == ["review"]
        assert fake.destroyed == ["review"]


class TestActionLabel:
    @pytest.mark.parametrize(
        "action, names, expected",
        [
            (0, ("w",), "noop"),
            (1, ("w", "a"), "w"),
            (np.int64(2), ("w", "a"), "a"),
            (3, ("w", "a"), "3"),  # out of range falls back to the index
            (1, None, "1"),
            ("jump", ("w",), "jump"),  # non-numeric actions are shown as-is
            (None, ("w",), "None"),
        ],
    )
    def test_labels(self, action, names, expected):
        assert trajectory_store._action_label(action, names) == expected


def test_import_cv2_returns_the_module_when_installed(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(sys.modules, "cv2", sentinel)

    assert trajectory_store._import_cv2() is sentinel
