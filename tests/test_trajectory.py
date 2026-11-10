import pytest

from copy_cat.data.trajectory import Trajectory


@pytest.fixture
def trajectory():
    """A three-step trajectory with distinct states, actions, rewards and info."""
    trajectory = Trajectory()

    for step in range(3):
        trajectory.record_step(
            state=f"s{step}", action=step, reward=float(step), timestamp=step / 10
        )

    return trajectory


class TestRecording:
    def test_starts_empty(self):
        trajectory = Trajectory()

        assert trajectory.get_trajectory_length() == 0
        assert trajectory.state_actions == []
        assert trajectory.rewards == []
        assert trajectory.additional_info == []

    def test_record_step_appends_parallel_entries(self, trajectory):
        assert trajectory.get_trajectory_length() == 3
        assert trajectory.state_actions == [("s0", 0), ("s1", 1), ("s2", 2)]
        assert trajectory.rewards == [0.0, 1.0, 2.0]
        assert trajectory.additional_info[1] == {"timestamp": 0.1}

    def test_reward_defaults_to_none(self):
        trajectory = Trajectory()
        trajectory.record_step(state="s", action=1)

        assert trajectory.rewards == [None]
        assert trajectory.additional_info == [{}]

    def test_clear_resets_everything(self, trajectory):
        trajectory.clear()

        assert trajectory.get_trajectory_length() == 0
        assert trajectory.rewards == []
        assert trajectory.additional_info == []


class TestUpdateStep:
    def test_replaces_an_existing_step(self, trajectory):
        trajectory.update_step("new", 9, 5.0, timestep=1, note="edited")

        assert trajectory.state_actions[1] == ("new", 9)
        assert trajectory.rewards[1] == 5.0
        assert trajectory.additional_info[1] == {"note": "edited"}
        assert trajectory.get_trajectory_length() == 3

    @pytest.mark.parametrize("timestep", [None, 3])
    def test_appends_when_timestep_is_none_or_the_next_index(self, trajectory, timestep):
        trajectory.update_step("s3", 3, 3.0, timestep=timestep)

        assert trajectory.get_trajectory_length() == 4
        assert trajectory.state_actions[3] == ("s3", 3)
        assert trajectory.rewards[3] == 3.0

    @pytest.mark.xfail(
        strict=True,
        reason="Known bug: update_step forwards kwargs=kwargs, nesting the info dict.",
    )
    def test_appending_keeps_info_flat(self, trajectory):
        trajectory.update_step("s3", 3, 3.0, note="appended")

        assert trajectory.additional_info[3] == {"note": "appended"}

    @pytest.mark.parametrize("timestep", [-1, 4])
    def test_rejects_out_of_range_timesteps(self, trajectory, timestep):
        with pytest.raises(AssertionError):
            trajectory.update_step("x", 0, 0.0, timestep=timestep)


class TestAccessors:
    def test_get_state_and_state_action(self, trajectory):
        assert trajectory.get_state(2) == "s2"
        assert trajectory.get_state_action(0) == ("s0", 0)

    def test_get_reward_is_one_indexed(self, trajectory):
        assert trajectory.get_reward(1) == 0.0
        assert trajectory.get_reward(3) == 2.0

    def test_get_step_pairs_a_state_with_its_reward(self, trajectory):
        assert trajectory.get_step(1) == ("s1", 1, 1.0)

    def test_get_step_with_info(self, trajectory):
        assert trajectory.get_step_with_info(2) == ("s2", 2, 2.0, {"timestamp": 0.2})

    @pytest.mark.parametrize(
        "accessor, timestep",
        [
            ("get_state", -1),
            ("get_state", 3),
            ("get_state_action", 3),
            ("get_step", 3),
            ("get_reward", 0),
            ("get_reward", 4),
        ],
    )
    def test_out_of_range_access_is_rejected(self, trajectory, accessor, timestep):
        with pytest.raises(AssertionError):
            getattr(trajectory, accessor)(timestep)

    def test_appearance_before_timestep_only_looks_at_the_prefix(self, trajectory):
        assert trajectory.check_state_action_appearance_before_timestep(("s1", 1), 2)
        assert not trajectory.check_state_action_appearance_before_timestep(("s1", 1), 1)
        assert not trajectory.check_state_action_appearance_before_timestep(("s9", 9), 3)
