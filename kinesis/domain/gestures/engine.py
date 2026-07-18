"""Runs all registered recognizers each frame and arbitrates them into GestureEvents.

Recognizers are single-hand and stateful, so the engine keeps a fresh set per
hand (keyed by handedness), built from the given factories. `update` is called
once per detected hand per frame.
"""
from collections.abc import Callable, Iterable

from kinesis.domain.gestures.base import Recognizer
from kinesis.domain.types import GestureEvent, HandFeatures


class GestureEngine:
    def __init__(self, factories: Iterable[Callable[[], Recognizer]]):
        self._factories = list(factories)
        self._per_hand: dict[str, list[Recognizer]] = {}

    def update(self, features: HandFeatures, hand: str) -> list[GestureEvent]:
        recognizers = self._per_hand.get(hand)
        if recognizers is None:
            recognizers = [make() for make in self._factories]
            self._per_hand[hand] = recognizers

        events: list[GestureEvent] = []
        for recognizer in recognizers:
            events.extend(recognizer.update(features, hand))
        return events

    def forget(self, hand: str) -> None:
        """Reset a hand's recognizers when it leaves the frame."""
        self._per_hand.pop(hand, None)

    def reset(self) -> None:
        self._per_hand.clear()
