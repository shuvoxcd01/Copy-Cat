from typing import Optional, Tuple

import mss
import numpy as np

from copy_cat.observation_capture.observation_capture import ObservationCapture


class ScreenObservationCapture(ObservationCapture):
    """Concrete implementation of ObservationCapture for screen observations."""

    def __init__(
        self,
        monitor: int = 1,
        region: Optional[Tuple[int, int, int, int]] = None,
    ):
        if monitor < 0:
            raise ValueError("Monitor index must be non-negative.")

        if region is not None:
            left, top, width, height = region
            if width <= 0 or height <= 0:
                raise ValueError("Region width and height must be positive.")

        self.monitor = monitor
        self.region = region
        self._sct = None

    def capture_observation(self) -> np.ndarray:
        """Capture the current screen observation as an RGB array."""
        if self._sct is None:
            self._sct = mss.MSS()

        screenshot = self._sct.grab(self._get_grab_area())

        # mss returns BGRA; drop the alpha channel and reverse to RGB.
        frame = np.asarray(screenshot)[:, :, :3][:, :, ::-1]

        return np.ascontiguousarray(frame)

    def _get_grab_area(self):
        """Return the area to grab, as expected by mss."""
        if self.region is None:
            return self._sct.monitors[self.monitor]

        left, top, width, height = self.region

        return {"left": left, "top": top, "width": width, "height": height}

    def close(self):
        """Release the underlying screen capture resources."""
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager and release resources."""
        self.close()


if __name__ == "__main__":
    # Example usage
    with ScreenObservationCapture(monitor=1) as screen_capture:
        observation = screen_capture.capture_observation()
        print("Captured observation shape:", observation.shape)

        # Display observation using OpenCV (optional)
        import cv2
        cv2.imshow("Screen Observation", observation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
