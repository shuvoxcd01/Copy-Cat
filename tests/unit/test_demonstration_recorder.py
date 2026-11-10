import pytest
from pynput import keyboard

from copy_cat.action_capture.action_capture import ActionCapture
from copy_cat.observation_capture.observation_capture import ObservationCapture
from copy_cat.recorder import demonstration_recorder
from copy_cat.recorder.demonstration_recorder import DemonstrationRecorder
from fakes import char, frame

TOGGLE = keyboard.Key.f9


class ScriptedActions(ActionCapture):
    """Returns a fixed sequence of actions; the first poll is the pre-session flush."""

    def __init__(self, actions, keys=("w", "a"), noop=0):
        self.actions = list(actions)
        self.keys = keys
        self.noop = noop
        self.polls = 0

    @property
    def noop_action(self):
        return self.noop

    def capture_action(self):
        self.polls += 1

        return self.actions.pop(0) if self.actions else 0


class ScriptedObservations(ObservationCapture):
    """Returns numbered frames and presses the toggle key after a set number of them.

    cost is how long each grab takes on the fake clock, to simulate a slow capture.
    """

    def __init__(self, stop_after, listener_source, clock, cost=0.0, fail_at=None):
        self.stop_after = stop_after
        self.listener_source = listener_source
        self.clock = clock
        self.cost = cost
        self.fail_at = fail_at
        self.count = 0
        self.closed = False

    def capture_observation(self):
        self.count += 1

        if self.count == self.fail_at:
            raise KeyboardInterrupt

        self.clock.advance(self.cost)

        if self.count == self.stop_after:
            self.listener_source.instances[-1].tap(TOGGLE)

        return frame(self.count)

    def close(self):
        self.closed = True


@pytest.fixture
def session(fake_listener, fake_clock, monkeypatch):
    """Patch time and the toggle listener so record() starts at once and runs instantly."""
    monkeypatch.setattr(demonstration_recorder, "time", fake_clock)
    fake_listener.on_start = lambda listener: listener.tap(TOGGLE)

    def build(actions, stop_after, cost=0.0, fail_at=None, **kwargs):
        observations = ScriptedObservations(
            stop_after, fake_listener, fake_clock, cost=cost, fail_at=fail_at
        )
        recorder = DemonstrationRecorder(
            observations, ScriptedActions(actions), verbose=False, **kwargs
        )

        return recorder

    return build


def frame_ids(trajectory):
    """Return which scripted frame each step stored."""
    return [
        int(trajectory.get_state(step)[0, 0, 0])
        for step in range(trajectory.get_trajectory_length())
    ]


class TestConstruction:
    def test_normalizes_the_toggle_key(self):
        recorder = DemonstrationRecorder(None, ScriptedActions([]), toggle_key=" F10 ")

        assert recorder.toggle_key == "f10"
        assert recorder.last_trajectory is None

    def test_rejects_a_toggle_key_that_is_also_an_action(self):
        with pytest.raises(ValueError, match="also a tracked action key"):
            DemonstrationRecorder(None, ScriptedActions([], keys=("w", "f9")), toggle_key="F9")

    def test_rejects_unknown_toggle_keys(self):
        with pytest.raises(ValueError, match="Unknown key name"):
            DemonstrationRecorder(None, ScriptedActions([]), toggle_key="nope")

    def test_accepts_captures_without_a_key_list(self):
        class Bare:
            def capture_action(self):
                return 0

        DemonstrationRecorder(None, Bare())


class TestRecord:
    def test_pairs_each_frame_with_the_following_action(self, session):
        # The first scripted action is consumed by the pre-session flush.
        recorder = session(actions=[9, 1, 2, 0], stop_after=3)

        trajectory = recorder.record(fps=10)

        assert frame_ids(trajectory) == [1, 2, 3]
        assert [action for _, action in trajectory.state_actions] == [1, 2, 0]
        assert trajectory.rewards == [None, None, None]
        assert recorder.action_capture.polls == 4

    def test_records_timestamps_relative_to_the_start(self, session):
        trajectory = session(actions=[0, 1, 1, 1], stop_after=3).record(fps=10)

        stamps = [info["timestamp"] for info in trajectory.additional_info]
        assert stamps == pytest.approx([0.1, 0.2, 0.3])

    def test_sleeps_only_the_remainder_of_each_period(self, session, fake_clock):
        session(actions=[0, 1, 1], stop_after=2, cost=0.03).record(fps=10)

        assert fake_clock.sleeps == pytest.approx([0.07, 0.07])

    def test_resyncs_instead_of_catching_up_when_behind(self, session, fake_clock):
        trajectory = session(actions=[0, 1, 1, 1], stop_after=3, cost=0.25).record(fps=10)

        # Never sleeps, and never bursts to make up for the lost time.
        assert fake_clock.sleeps == []
        stamps = [info["timestamp"] for info in trajectory.additional_info]
        assert stamps == pytest.approx([0.25, 0.5, 0.75])

    def test_sets_action_names_and_last_trajectory(self, session):
        recorder = session(actions=[0, 1], stop_after=1)

        trajectory = recorder.record()

        assert trajectory.action_names == ("w", "a")
        assert recorder.last_trajectory is trajectory

    def test_stopping_immediately_gives_one_step(self, session):
        assert session(actions=[0, 2], stop_after=1).record().get_trajectory_length() == 1

    def test_strip_noop_drops_the_frame_too(self, session):
        recorder = session(actions=[0, 1, 0, 0, 2], stop_after=4, strip_noop=True)

        trajectory = recorder.record(fps=10)

        assert [action for _, action in trajectory.state_actions] == [1, 2]
        assert frame_ids(trajectory) == [1, 4]
        stamps = [info["timestamp"] for info in trajectory.additional_info]
        assert stamps == pytest.approx([0.1, 0.4])

    def test_strip_noop_uses_the_capture_noop_value(self, session):
        recorder = session(actions=[0, 0, -1, 1], stop_after=3, strip_noop=True)
        recorder.action_capture.noop = -1

        trajectory = recorder.record()

        assert [action for _, action in trajectory.state_actions] == [0, 1]

    @pytest.mark.parametrize("fps", [0, -5])
    def test_rejects_non_positive_fps(self, session, fake_listener, fps):
        with pytest.raises(ValueError, match="fps must be positive"):
            session(actions=[], stop_after=1).record(fps=fps)

        assert fake_listener.instances == []

    def test_can_record_repeatedly_with_the_same_captures(self, session, fake_listener):
        recorder = session(actions=[0, 1, 0, 2], stop_after=1)

        first = recorder.record()
        recorder.observation_capture.stop_after = 2
        second = recorder.record()

        assert first is not second
        assert [action for _, action in second.state_actions] == [2]
        assert len(fake_listener.instances) == 2
        assert not recorder.observation_capture.closed


class TestListener:
    def test_listener_is_stopped_after_recording(self, session, fake_listener):
        recorder = session(actions=[0, 1], stop_after=1)

        recorder.record()

        listener = fake_listener.instances[0]
        assert listener.suppress is False
        assert listener.stopped and listener.joined
        assert recorder._listener is None

    def test_interrupted_session_keeps_partial_steps(self, session, fake_listener):
        recorder = session(actions=[0, 1, 2], stop_after=99, fail_at=3)

        with pytest.raises(KeyboardInterrupt):
            recorder.record()

        assert recorder.last_trajectory.get_trajectory_length() == 2
        assert fake_listener.instances[0].stopped

    def test_start_listener_is_idempotent(self, fake_listener):
        recorder = DemonstrationRecorder(None, ScriptedActions([]))

        recorder._start_toggle_listener()
        recorder._start_toggle_listener()

        assert len(fake_listener.instances) == 1

    def test_stop_without_start_is_harmless(self):
        DemonstrationRecorder(None, ScriptedActions([]))._stop_toggle_listener()


class TestToggleDetection:
    @pytest.fixture
    def recorder(self):
        return DemonstrationRecorder(None, ScriptedActions([]), toggle_key="9")

    def test_auto_repeat_fires_once_until_released(self, recorder):
        recorder._on_toggle_press(char("9"))
        recorder._toggle.clear()

        recorder._on_toggle_press(char("9"))
        assert not recorder._toggle.is_set()

        recorder._on_toggle_release(char("9"))
        recorder._on_toggle_press(char("9"))
        assert recorder._toggle.is_set()

    def test_other_keys_do_not_toggle(self, recorder):
        recorder._on_toggle_press(char("8"))
        recorder._on_toggle_press(keyboard.Key.f9)
        recorder._on_toggle_press(keyboard.KeyCode(vk=0))

        assert not recorder._toggle.is_set()

    def test_release_of_other_keys_keeps_the_hold(self, recorder):
        recorder._on_toggle_press(char("9"))
        recorder._toggle.clear()
        recorder._on_toggle_release(char("8"))

        recorder._on_toggle_press(char("9"))

        assert not recorder._toggle.is_set()

    def test_special_toggle_key_matches_its_key_member(self):
        recorder = DemonstrationRecorder(None, ScriptedActions([]), toggle_key="f9")

        assert recorder._is_toggle(keyboard.Key.f9)
        assert not recorder._is_toggle(keyboard.Key.f10)
        assert not recorder._is_toggle(char("f"))

    def test_character_toggle_is_case_insensitive(self):
        recorder = DemonstrationRecorder(None, ScriptedActions([]), toggle_key="q")

        assert recorder._is_toggle(char("Q"))


class TestLogging:
    def test_verbose_reports_the_session(self, session, capsys):
        recorder = session(actions=[0, 1, 0], stop_after=2, strip_noop=True)
        recorder.verbose = True

        recorder.record(fps=10)

        out = capsys.readouterr().out
        assert "Press f9 to start recording" in out
        assert "Recording at 10 Hz" in out
        assert "Stopped: 1 steps" in out
        assert "1 no-op steps dropped" in out

    def test_quiet_mode_omits_session_messages(self, session, capsys):
        session(actions=[0, 1], stop_after=1).record()

        out = capsys.readouterr().out
        assert "Press f9" not in out
        assert "Stopped" not in out

    @pytest.mark.xfail(
        strict=True, reason="Known bug: the per-step print ignores verbose=False."
    )
    def test_quiet_mode_prints_nothing(self, session, capsys):
        session(actions=[0, 1], stop_after=1).record()

        assert capsys.readouterr().out == ""

    def test_achieved_rate_is_zero_when_no_time_passes(
        self, fake_listener, fake_clock, monkeypatch, capsys
    ):
        # A clock that never moves makes elapsed 0; the summary must not divide by it.
        monkeypatch.setattr(fake_clock, "sleep", lambda seconds: None)
        monkeypatch.setattr(demonstration_recorder, "time", fake_clock)
        fake_listener.on_start = lambda listener: listener.tap(TOGGLE)
        observations = ScriptedObservations(1, fake_listener, fake_clock)

        DemonstrationRecorder(observations, ScriptedActions([0, 1])).record()

        assert "0.0 Hz achieved" in capsys.readouterr().out
