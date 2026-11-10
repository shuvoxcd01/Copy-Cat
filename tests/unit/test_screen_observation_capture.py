import numpy as np
import pytest

from copy_cat.observation_capture import screen_observation_capture
from copy_cat.observation_capture.screen_observation_capture import ScreenObservationCapture

# A 1x2 BGRA image: a pure-blue pixel then a pure-red one, with alpha set.
BGRA = np.array([[[255, 0, 0, 255], [0, 0, 255, 255]]], dtype=np.uint8)


class FakeMSS:
    """Stands in for mss.MSS so no real screen is grabbed."""

    instances = []
    monitors = [
        {"left": 0, "top": 0, "width": 3000, "height": 1000},
        {"left": 0, "top": 0, "width": 1920, "height": 1000},
        {"left": 1920, "top": 0, "width": 1080, "height": 1000},
    ]

    def __init__(self):
        self.grabs = []
        self.closed = False
        FakeMSS.instances.append(self)

    def grab(self, area):
        self.grabs.append(area)

        return BGRA.copy()

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_mss(monkeypatch):
    FakeMSS.instances = []
    monkeypatch.setattr(screen_observation_capture.mss, "MSS", FakeMSS)

    return FakeMSS


class TestConstruction:
    def test_defaults(self):
        capture = ScreenObservationCapture()

        assert capture.monitor == 1
        assert capture.region is None

    def test_rejects_negative_monitor(self):
        with pytest.raises(ValueError, match="non-negative"):
            ScreenObservationCapture(monitor=-1)

    @pytest.mark.parametrize("region", [(0, 0, 0, 10), (0, 0, 10, -1)])
    def test_rejects_empty_regions(self, region):
        with pytest.raises(ValueError, match="positive"):
            ScreenObservationCapture(region=region)

    def test_opens_mss_lazily(self, fake_mss):
        ScreenObservationCapture()

        assert fake_mss.instances == []


class TestCapture:
    def test_returns_contiguous_rgb_uint8(self):
        observation = ScreenObservationCapture().capture_observation()

        assert observation.shape == (1, 2, 3)
        assert observation.dtype == np.uint8
        assert observation.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(observation[0, 0], [0, 0, 255])  # blue
        np.testing.assert_array_equal(observation[0, 1], [255, 0, 0])  # red

    def test_grabs_the_requested_monitor(self, fake_mss):
        ScreenObservationCapture(monitor=2).capture_observation()

        assert fake_mss.instances[0].grabs == [FakeMSS.monitors[2]]

    def test_region_overrides_the_monitor(self, fake_mss):
        ScreenObservationCapture(monitor=2, region=(10, 20, 30, 40)).capture_observation()

        assert fake_mss.instances[0].grabs == [
            {"left": 10, "top": 20, "width": 30, "height": 40}
        ]

    def test_reuses_one_mss_handle(self, fake_mss):
        capture = ScreenObservationCapture()

        capture.capture_observation()
        capture.capture_observation()

        assert len(fake_mss.instances) == 1
        assert len(fake_mss.instances[0].grabs) == 2


class TestClose:
    def test_close_releases_the_handle(self, fake_mss):
        capture = ScreenObservationCapture()
        capture.capture_observation()

        capture.close()

        assert fake_mss.instances[0].closed
        assert capture._sct is None

    def test_close_before_capture_is_harmless(self):
        ScreenObservationCapture().close()

    def test_capture_after_close_reopens(self, fake_mss):
        capture = ScreenObservationCapture()
        capture.capture_observation()
        capture.close()

        capture.capture_observation()

        assert len(fake_mss.instances) == 2

    def test_context_manager_closes(self, fake_mss):
        with ScreenObservationCapture() as capture:
            capture.capture_observation()

        assert fake_mss.instances[0].closed
