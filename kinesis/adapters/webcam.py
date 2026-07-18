"""WebcamSource(FrameSource): wraps cv2.VideoCapture, yields Frame (BGR). Capture runs on a background thread, keeping only the latest frame."""
import threading
import time

import cv2

from kinesis.domain.types import Frame
from kinesis.ports.frame_source import FrameSource


class WebcamSource(FrameSource):
    """Reads mirrored frames from a webcam as domain Frames (BGR, as OpenCV gives).

    Capture runs on a background thread that keeps only the NEWEST frame, so the
    pipeline always processes the freshest image and stale frames can't pile up
    into cursor lag. read() blocks until a fresh frame is available. cv2 is sealed
    inside this file; colour conversion to RGB is the tracker's job.
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, mirror: bool = True, fps: int = 60):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open webcam at index {index} (check camera permission)")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)  # a request — the camera may cap lower (see scripts/probe_camera.py)
        self._mirror = mirror
        self._frame_id = 0
        self._t0 = time.perf_counter()
        self._latest = None
        self._running = True
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ok, image = self._cap.read()
            if not ok:
                continue
            if self._mirror:
                image = cv2.flip(image, 1)
            with self._cond:
                self._latest = image  # overwrite: only the newest frame survives
                self._cond.notify()

    def read(self) -> Frame | None:
        with self._cond:
            while self._latest is None and self._running:
                self._cond.wait(timeout=1.0)
            image = self._latest
            self._latest = None
        if image is None:
            return None
        frame = Frame(image=image, timestamp=time.perf_counter() - self._t0, frame_id=self._frame_id)
        self._frame_id += 1
        return frame

    def close(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)
        self._cap.release()
