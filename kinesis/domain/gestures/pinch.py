"""Pinch recognizer (thumb-tip to index-tip): emits PINCH began/changed/ended."""
from kinesis.domain.gestures.base import Recognizer
from kinesis.domain.types import GestureEvent, GesturePhase, GestureType, HandFeatures

MIDDLE = 2  # index into fingers_extended (thumb, index, middle, ring, pinky)


class PinchRecognizer(Recognizer):
    """Pinch = thumb and index brought together, WITH the hand open.

    Two guards, both learned from real recordings:
      * open-hand gate — the pinch only counts when the middle finger is extended.
        A fist-with-index-out (pointing) curls the middle finger, so tucking the
        thumb into the palm while pointing can't fake a click.
      * debounce — the pinch condition must hold for `hold_frames` consecutive
        frames (~100ms at 30fps) before firing, so a momentary dip is ignored.
    Release uses hysteresis: the gap must reopen past `off` (> `on`) to end.
    """

    def __init__(self, on: float = 0.35, off: float = 0.5, hold_frames: int = 3):
        self.on = on
        self.off = off
        self.hold_frames = hold_frames
        self._pinched = False
        self._candidate = 0  # consecutive frames meeting the engage condition

    def update(self, features: HandFeatures, hand: str) -> list[GestureEvent]:
        d = features.pinch_distance
        open_hand = bool(features.fingers_extended[MIDDLE])

        if self._pinched:
            if d > self.off:
                self._pinched = False
                self._candidate = 0
                return [self._event(GesturePhase.ENDED, features, hand)]
            return [self._event(GesturePhase.CHANGED, features, hand)]

        if open_hand and d < self.on:
            self._candidate += 1
            if self._candidate >= self.hold_frames:
                self._pinched = True
                self._candidate = 0
                return [self._event(GesturePhase.BEGAN, features, hand)]
        else:
            self._candidate = 0
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
        self._candidate = 0
