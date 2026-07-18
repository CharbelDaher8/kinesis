"""FrameSource port: read() -> Frame or None. Implemented by adapters/webcam.py.

This is the interface only — no cv2 here. The concrete camera implementation (and
its cv2 import) lives in adapters/webcam.py; a RecordingSource for replay is
another implementation of the same port.
"""
from abc import ABC, abstractmethod

from kinesis.domain.types import Frame


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next Frame, or None when the stream ends / a read fails."""

    def close(self) -> None:
        """Release resources. Override if needed."""
