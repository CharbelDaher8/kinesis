"""WebcamSource(FrameSource): wraps cv2.VideoCapture, yields Frame (BGR). Thread + keep-latest is a later optimization."""
import time

import cv2

from kinesis.domain.types import Frame
from kinesis.ports.frame_source import FrameSource


class WebcamSource(FrameSource):
    """Reads mirrored frames from a webcam as domain Frames (BGR, as OpenCV gives).

    cv2 is sealed inside this file — nothing else in the app imports it. Colour
    conversion to RGB is the tracker's job, not the camera's.
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, mirror: bool = True):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open webcam at index {index} (check camera permission)")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._mirror = mirror
        self._frame_id = 0
        self._t0 = time.perf_counter()

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok:
            return None
        if self._mirror:
            image = cv2.flip(image, 1)
        frame = Frame(image=image, timestamp=time.perf_counter() - self._t0, frame_id=self._frame_id)
        self._frame_id += 1
        return frame

    def close(self) -> None:
        self._cap.release()
