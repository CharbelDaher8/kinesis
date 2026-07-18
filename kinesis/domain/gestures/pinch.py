"""Pinch recognizer (thumb-tip to index-tip): emits PINCH began/changed/ended."""
from kinesis.domain.gestures.base import Recognizer
from kinesis.domain.types import GestureEvent, GesturePhase, GestureType, HandFeatures


class PinchRecognizer(Recognizer):
    """Pinch = thumb and index brought together.

    Hysteresis: engage when pinch_distance drops below `on`, release only when it
    rises above `off` (off > on). The gap between the two thresholds stops a value
    hovering near the boundary from flickering pinched/unpinched.
    """

    def __init__(self, on: float = 0.35, off: float = 0.5):
        self.on = on
        self.off = off
        self._pinched = False

    def update(self, features: HandFeatures, hand: str) -> list[GestureEvent]:
        d = features.pinch_distance
        if not self._pinched and d < self.on:
            self._pinched = True
            return [self._event(GesturePhase.BEGAN, features, hand)]
        if self._pinched and d > self.off:
            self._pinched = False
            return [self._event(GesturePhase.ENDED, features, hand)]
        if self._pinched:
            return [self._event(GesturePhase.CHANGED, features, hand)]
        return []

    def _event(self, phase: GesturePhase, features: HandFeatures, hand: str) -> GestureEvent:
        return GestureEvent(
            type=GestureType.PINCH,
            phase=phase,
            hand=hand,
            timestamp=features.timestamp,
            data={"point": features.cursor_point},
        )

    def reset(self) -> None:
        self._pinched = False
