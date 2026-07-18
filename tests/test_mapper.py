"""Unit tests for the intent mapper — pure.

Run: PYTHONPATH=. .venv/bin/python tests/test_mapper.py
"""
from kinesis.domain.intent.mapper import IntentMapper
from kinesis.domain.types import (
    GestureEvent,
    GesturePhase,
    GestureType,
    IntentType,
)


def test_point_maps_to_move_cursor():
    m = IntentMapper()
    ev = GestureEvent(GestureType.POINT, GesturePhase.CHANGED, "Right", 1.0, {"point": (0.3, 0.6)})
    it = m.map(ev)
    assert it.type is IntentType.MOVE_CURSOR
    assert it.params == {"x": 0.3, "y": 0.6}
    assert it.timestamp == 1.0


def test_pinch_began_maps_to_click():
    m = IntentMapper()
    ev = GestureEvent(GestureType.PINCH, GesturePhase.BEGAN, "Right", 2.0, {"point": (0.5, 0.5)})
    assert m.map(ev).type is IntentType.CLICK


def test_unmapped_gesture_returns_none():
    m = IntentMapper()
    ev = GestureEvent(GestureType.PINCH, GesturePhase.ENDED, "Right", 3.0)
    assert m.map(ev) is None


def test_swipe_direction_maps_to_pages():
    m = IntentMapper()
    left = GestureEvent(GestureType.SWIPE, GesturePhase.RECOGNIZED, "Right", 0.0, {"direction": "left"})
    right = GestureEvent(GestureType.SWIPE, GesturePhase.RECOGNIZED, "Right", 0.0, {"direction": "right"})
    assert m.map(left).type is IntentType.PREV_PAGE
    assert m.map(right).type is IntentType.NEXT_PAGE


def test_context_profile_overrides_default():
    m = IntentMapper(profiles={"Reader": {}})  # Reader: pinch does nothing
    ev = GestureEvent(GestureType.PINCH, GesturePhase.BEGAN, "Right", 1.0, {"point": (0.5, 0.5)})
    assert m.map(ev, context="Reader") is None
    assert m.map(ev, context="Anything else").type is IntentType.CLICK  # default fallback


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _run()
