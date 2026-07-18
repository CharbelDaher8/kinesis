"""Unit tests for domain.features — pure math on synthetic landmarks, no camera.

Run: .venv/bin/python tests/test_features.py
"""
import numpy as np

from kinesis.domain.features import (
    FeatureExtractor,
    cursor_point,
    fingers_extended,
    pinch_distance,
)
from kinesis.domain.types import HandObservation

# A plausible open right hand, fingers pointing up (y decreases upward).
OPEN = np.array([
    [0.50, 0.90, 0.0],  # 0  wrist
    [0.42, 0.82, 0.0],  # 1
    [0.37, 0.74, 0.0],  # 2  thumb mcp
    [0.33, 0.68, 0.0],  # 3
    [0.30, 0.63, 0.0],  # 4  thumb tip
    [0.45, 0.60, 0.0],  # 5  index mcp
    [0.44, 0.48, 0.0],  # 6  index pip
    [0.43, 0.40, 0.0],  # 7
    [0.43, 0.33, 0.0],  # 8  index tip
    [0.52, 0.58, 0.0],  # 9  middle mcp
    [0.52, 0.45, 0.0],  # 10 middle pip
    [0.52, 0.36, 0.0],  # 11
    [0.52, 0.28, 0.0],  # 12 middle tip
    [0.58, 0.60, 0.0],  # 13 ring mcp
    [0.59, 0.47, 0.0],  # 14 ring pip
    [0.59, 0.39, 0.0],  # 15
    [0.59, 0.32, 0.0],  # 16 ring tip
    [0.64, 0.64, 0.0],  # 17 pinky mcp
    [0.66, 0.54, 0.0],  # 18 pinky pip
    [0.67, 0.47, 0.0],  # 19
    [0.68, 0.42, 0.0],  # 20 pinky tip
])


def _pinched():
    lm = OPEN.copy()
    lm[4] = [0.43, 0.35, 0.0]  # thumb tip brought onto the index tip
    return lm


def _fist():
    lm = OPEN.copy()
    lm[8] = [0.45, 0.62, 0.0]   # tips curled back toward the palm/wrist
    lm[12] = [0.52, 0.62, 0.0]
    lm[16] = [0.58, 0.62, 0.0]
    lm[20] = [0.63, 0.64, 0.0]
    return lm


def test_cursor_point_is_index_tip():
    x, y = cursor_point(OPEN)
    assert abs(x - 0.43) < 1e-9 and abs(y - 0.33) < 1e-9


def test_pinch_distance_drops_when_pinched():
    open_d, pinched_d = pinch_distance(OPEN), pinch_distance(_pinched())
    assert open_d > 0.8
    assert pinched_d < 0.3
    assert pinched_d < open_d


def test_pinch_distance_is_scale_invariant():
    # scale the whole hand about the origin (== moving farther) -> ratio unchanged
    assert abs(pinch_distance(OPEN) - pinch_distance(OPEN * 0.5)) < 1e-9


def test_fingers_extended_on_open_hand():
    assert fingers_extended(OPEN) == (True, True, True, True, True)


def test_fingers_curled_in_fist():
    _, *four = fingers_extended(_fist())
    assert four == [False, False, False, False]


def test_velocity_zero_then_tracks_motion():
    fx = FeatureExtractor()
    f1 = fx.extract(HandObservation(OPEN, "Right", 0.0))
    assert f1.velocity == (0.0, 0.0)  # first frame: no history
    moved = OPEN + np.array([0.1, 0.0, 0.0])
    f2 = fx.extract(HandObservation(moved, "Right", 0.1))
    assert abs(f2.velocity[0] - 1.0) < 1e-6  # +0.1 over 0.1s
    assert abs(f2.velocity[1] - 0.0) < 1e-6


def test_velocity_is_tracked_per_hand():
    fx = FeatureExtractor()
    fx.extract(HandObservation(OPEN, "Right", 0.0))
    f = fx.extract(HandObservation(OPEN, "Left", 0.05))  # different hand starts fresh
    assert f.velocity == (0.0, 0.0)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _run()
