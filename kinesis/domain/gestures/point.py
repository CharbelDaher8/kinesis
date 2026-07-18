"""Pointing recognizer: index-finger position used to drive the cursor."""
from kinesis.domain.gestures.base import Recognizer
from kinesis.domain.types import GestureEvent, GesturePhase, GestureType, HandFeatures


class PointRecognizer(Recognizer):
    """Pointing pose = index extended while the middle finger is curled.

    Emits POINT began on entering the pose, changed every frame while held (so the
    cursor tracks the fingertip), and ended on leaving it. The cursor position rides
    in data["point"].
    """

    def __init__(self):
        self._pointing = False

    def update(self, features: HandFeatures, hand: str) -> list[GestureEvent]:
        _thumb, index, middle, _ring, _pinky = features.fingers_extended
        pose = index and not middle

        if pose and not self._pointing:
            self._pointing = True
            return [self._event(GesturePhase.BEGAN, features, hand)]
        if pose and self._pointing:
            return [self._event(GesturePhase.CHANGED, features, hand)]
        if not pose and self._pointing:
            self._pointing = False
            return [self._event(GesturePhase.ENDED, features, hand)]
        return []

    def _event(self, phase: GesturePhase, features: HandFeatures, hand: str) -> GestureEvent:
        return GestureEvent(
            type=GestureType.POINT,
            phase=phase,
            hand=hand,
            timestamp=features.timestamp,
            data={"point": features.cursor_point},
        )

    def reset(self) -> None:
        self._pointing = False
