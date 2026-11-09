import torch

from copy_cat.policies.base_policy import BasePolicy


class ExamplePolicy(BasePolicy):
    def __init__(self, value=7):
        super().__init__()
        self.value = value

    def get_action(self, state):
        return 0

    def get_action_prob(self, state, action):
        return 1.0

    def get_all_action_probabilities(self, states):
        return torch.ones(len(states), 1)

    def update(self, state, action):
        return None


def test_policy_round_trip_saves_and_loads(tmp_path):
    original = ExamplePolicy(value=42)
    path = tmp_path / "example_policy.pt"

    original.save(path)
    loaded = ExamplePolicy.load(path)

    assert isinstance(loaded, ExamplePolicy)
    assert loaded.value == 42
    assert loaded.get_action(None) == 0
