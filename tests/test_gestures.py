"""Unit tests for gesture recognizers + engine — synthetic HandFeatures, no camera.

Run: PYTHONPATH=. .venv/bin/python tests/test_gestures.py
"""
from kinesis.domain.gestures.engine import GestureEngine
from kinesis.domain.gestures.pinch import PinchRecognizer
from kinesis.domain.gestures.point import PointRecognizer
from kinesis.domain.types import GesturePhase, GestureType, HandFeatures


def feat(pinch=1.0, fingers=(False, False, False, False, False), point=(0.5, 0.5), t=0.0):
    """Build a HandFeatures with only the fields recognizers read; rest are filler."""
    return HandFeatures(
        cursor_point=point,
        pinch_distance=pinch,
        fingers_extended=fingers,
        centroid=(0.5, 0.5),
        velocity=(0.0, 0.0),
        palm_normal=(0.0, 0.0, 1.0),
        palm_angle=0.0,
        timestamp=t,
    )


def test_pinch_began_changed_ended():
    r = PinchRecognizer(on=0.35, off=0.5)
    assert r.update(feat(pinch=1.0), "Right") == []          # open: nothing
    (e,) = r.update(feat(pinch=0.2, t=1), "Right")
    assert e.type is GestureType.PINCH and e.phase is GesturePhase.BEGAN and e.hand == "Right"
    (e,) = r.update(feat(pinch=0.2, t=2), "Right")
    assert e.phase is GesturePhase.CHANGED
    (e,) = r.update(feat(pinch=0.6, t=3), "Right")
    assert e.phase is GesturePhase.ENDED


def test_pinch_hysteresis_does_not_flicker():
    r = PinchRecognizer(on=0.35, off=0.5)
    r.update(feat(pinch=0.2), "Right")                       # -> pinched
    (e,) = r.update(feat(pinch=0.45, t=1), "Right")          # between on and off
    assert e.phase is GesturePhase.CHANGED                    # stays pinched, no release


def test_pinch_carries_cursor_point():
    r = PinchRecognizer()
    (e,) = r.update(feat(pinch=0.1, point=(0.3, 0.7)), "Right")
    assert e.data["point"] == (0.3, 0.7)


def test_point_pose_begins_tracks_and_ends():
    r = PointRecognizer()
    (e,) = r.update(feat(fingers=(False, True, False, False, False), point=(0.3, 0.4)), "Right")
    assert e.type is GestureType.POINT and e.phase is GesturePhase.BEGAN
    assert e.data["point"] == (0.3, 0.4)
    (e,) = r.update(feat(fingers=(False, True, False, False, False), t=1), "Right")
    assert e.phase is GesturePhase.CHANGED
    (e,) = r.update(feat(fingers=(False, True, True, False, False), t=2), "Right")  # middle up -> not pointing
    assert e.phase is GesturePhase.ENDED


def test_engine_collects_events_and_stamps_hand():
    engine = GestureEngine([PinchRecognizer, PointRecognizer])
    events = engine.update(feat(pinch=0.2), "Right")         # pinch only (index not extended)
    assert any(e.type is GestureType.PINCH and e.phase is GesturePhase.BEGAN for e in events)
    assert all(e.hand == "Right" for e in events)


def test_engine_keeps_per_hand_state_separate():
    engine = GestureEngine([PinchRecognizer])
    engine.update(feat(pinch=0.2), "Right")                  # Right begins a pinch
    events = engine.update(feat(pinch=0.2), "Left")          # Left is a fresh recognizer
    assert any(e.phase is GesturePhase.BEGAN and e.hand == "Left" for e in events)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _run()
