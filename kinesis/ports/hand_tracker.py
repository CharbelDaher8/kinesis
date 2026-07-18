"""HandTracker port: Frame -> list of HandObservation. Implemented by adapters/mediapipe_tracker.py.

The interface only — no mediapipe here. Wrapping MediaPipe behind this port is
what lets the rest of the app stay ignorant of it, and lets tests feed recorded
observations instead of running the model.
"""
from abc import ABC, abstractmethod

from kinesis.domain.types import Frame, HandObservation


class HandTracker(ABC):
    @abstractmethod
    def track(self, frame: Frame) -> list[HandObservation]:
        """Detect hands in the frame; return zero or more HandObservations."""

    def close(self) -> None:
        """Release resources. Override if needed."""
