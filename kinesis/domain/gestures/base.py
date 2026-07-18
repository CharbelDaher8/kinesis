"""Recognizer base class for temporal gestures.

A Recognizer consumes ONE hand's HandFeatures stream and emits GestureEvents.
One instance handles a single hand — the engine builds a fresh set per hand — so
recognizers keep simple scalar state instead of per-hand dicts.

Recognizers *decide* (thresholds, hysteresis, phase transitions); features.py
only *measures*. Keep that split.
"""
from abc import ABC, abstractmethod

from kinesis.domain.types import GestureEvent, HandFeatures


class Recognizer(ABC):
    @abstractmethod
    def update(self, features: HandFeatures, hand: str) -> list[GestureEvent]:
        """Advance the recognizer by one frame; return events emitted (may be empty)."""

    def reset(self) -> None:
        """Drop any in-progress state (e.g. when tracking for this hand is lost)."""
