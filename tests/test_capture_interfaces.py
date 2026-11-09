import pytest

from copy_cat.action_capture.action_capture import ActionCapture
from copy_cat.observation_capture.observation_capture import ObservationCapture


@pytest.mark.parametrize("base", [ActionCapture, ObservationCapture])
def test_abstract_bases_cannot_be_instantiated(base):
    with pytest.raises(TypeError):
        base()


def test_action_capture_default_noop_is_zero():
    class Constant(ActionCapture):
        def capture_action(self):
            return 3

    capture = Constant()

    assert capture.noop_action == 0
    assert capture.capture_action() == 3


def test_deferring_to_the_abstract_action_method_raises():
    # A subclass that calls super() must get a clear error, not a silent None.
    class Deferring(ActionCapture):
        def capture_action(self):
            return super().capture_action()

    with pytest.raises(NotImplementedError):
        Deferring().capture_action()


def test_deferring_to_the_abstract_observation_method_raises():
    class Deferring(ObservationCapture):
        def capture_observation(self):
            return super().capture_observation()

    with pytest.raises(NotImplementedError):
        Deferring().capture_observation()
