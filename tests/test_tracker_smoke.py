"""Headless smoke test for the adapters: run the MediaPipe tracker on a synthetic
frame (no camera, no hand) and confirm the webcam module imports. Skips the
tracker if the model file is absent.

Run: PYTHONPATH=. .venv/bin/python tests/test_tracker_smoke.py
"""
import os

import numpy as np

from kinesis.domain.types import Frame

MODEL = "models/hand_landmarker.task"


def test_webcam_module_imports_without_opening_camera():
    import kinesis.adapters.webcam  # noqa: F401  — constructing would open the camera
    print("  ok  webcam module imports")


def test_tracker_runs_on_synthetic_frame():
    if not os.path.exists(MODEL):
        print("  skip tracker (model missing — run scripts/download_models.sh)")
        return
    from kinesis.adapters.mediapipe_tracker import MediaPipeTracker

    tracker = MediaPipeTracker()
    try:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)  # exercises the downscale path
        for i in range(3):
            hands = tracker.track(Frame(image=img, timestamp=i / 30.0, frame_id=i))
            assert isinstance(hands, list)  # noise -> no hand, but the whole path runs
    finally:
        tracker.close()
    print("  ok  tracker ran on synthetic frames")


def _run():
    test_webcam_module_imports_without_opening_camera()
    test_tracker_runs_on_synthetic_frame()
    print("done")


if __name__ == "__main__":
    _run()
