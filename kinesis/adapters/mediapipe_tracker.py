"""MediaPipeTracker(HandTracker): mediapipe.tasks HandLandmarker + models/hand_landmarker.task, RunningMode.VIDEO (strictly-increasing ms timestamps). Convert Frame BGR->RGB here; return HandObservation list. See scripts/hello_camera.py. Legacy mp.solutions is NOT available in 0.10.x."""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from kinesis.domain.types import Frame, HandObservation
from kinesis.ports.hand_tracker import HandTracker

DEFAULT_MODEL = "models/hand_landmarker.task"


class MediaPipeTracker(HandTracker):
    """Detects hands with MediaPipe Tasks (HandLandmarker, VIDEO mode).

    mediapipe is sealed inside this file. Returns domain HandObservations with
    normalized (21, 3) landmarks; converts BGR->RGB here since RGB is MediaPipe's
    requirement, not the camera's.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        num_hands: int = 1,  # one hand for the cursor keeps inference cheap and uniform
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.3,  # low, so tracking survives the occluded pointing fist
        proc_width: int = 640,  # downscale before inference; landmarks are normalized so accuracy holds
    ):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._proc_width = proc_width
        self._last_ts_ms = -1

    def track(self, frame: Frame) -> list[HandObservation]:
        image = frame.image
        if self._proc_width and image.shape[1] > self._proc_width:
            scale = self._proc_width / image.shape[1]
            image = cv2.resize(image, (self._proc_width, round(image.shape[0] * scale)))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._monotonic_ms(frame.timestamp))

        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            landmarks = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
            hands.append(
                HandObservation(
                    landmarks=landmarks,
                    handedness=handed[0].category_name,
                    timestamp=frame.timestamp,
                )
            )
        return hands

    def _monotonic_ms(self, timestamp: float) -> int:
        """detect_for_video needs strictly increasing millisecond timestamps."""
        ts_ms = int(timestamp * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        return ts_ms

    def close(self) -> None:
        self._detector.close()
