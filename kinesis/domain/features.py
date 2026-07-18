"""Derive HandFeatures from raw landmarks: pinch distance, finger states, palm
normal, centroid, velocity. Pure.

This is the geometry-to-meaning stage: 21 raw landmarks in, named signals out.
It is the ONLY place that knows landmark indices — recognizers downstream work
with pinch_distance / fingers_extended instead of raw points. It measures; it
does not decide (thresholds live in the gesture recognizers).
"""
import math

import numpy as np

from kinesis.domain.types import HandFeatures, HandObservation

# --- MediaPipe 21-landmark indices ---
WRIST = 0
THUMB_MCP = 2
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17

# (tip, pip) for the four non-thumb fingers, in HandFeatures order
FINGER_JOINTS = ((8, 6), (12, 10), (16, 14), (20, 18))  # index, middle, ring, pinky


def _xy(lm: np.ndarray, i: int) -> np.ndarray:
    return lm[i, :2]


def cursor_point(lm: np.ndarray) -> tuple[float, float]:
    """Index-fingertip position (landmark 8), normalized (x, y)."""
    return (float(lm[INDEX_TIP, 0]), float(lm[INDEX_TIP, 1]))


def pinch_distance(lm: np.ndarray) -> float:
    """Thumb-tip to index-tip gap, normalized by hand scale (wrist -> middle MCP)
    so it is invariant to how near/far the hand is from the camera."""
    gap = np.linalg.norm(_xy(lm, THUMB_TIP) - _xy(lm, INDEX_TIP))
    scale = np.linalg.norm(_xy(lm, WRIST) - _xy(lm, MIDDLE_MCP))
    if scale < 1e-6:
        return 0.0
    return float(gap / scale)


def fingers_extended(lm: np.ndarray) -> tuple[bool, bool, bool, bool, bool]:
    """(thumb, index, middle, ring, pinky) booleans.

    Four fingers: extended when the tip is farther from the wrist than its PIP
    joint (curling pulls the tip back toward the palm) — orientation-robust.
    Thumb: abducts sideways, so compare its distance from the pinky MCP against
    the thumb MCP's — a heuristic worth tuning against real recordings.
    """
    wrist = _xy(lm, WRIST)
    four = [
        np.linalg.norm(_xy(lm, tip) - wrist) > np.linalg.norm(_xy(lm, pip) - wrist)
        for tip, pip in FINGER_JOINTS
    ]
    pinky_mcp = _xy(lm, PINKY_MCP)
    thumb = np.linalg.norm(_xy(lm, THUMB_TIP) - pinky_mcp) > np.linalg.norm(
        _xy(lm, THUMB_MCP) - pinky_mcp
    )
    return (bool(thumb), *(bool(f) for f in four))


def centroid(lm: np.ndarray) -> tuple[float, float]:
    """Hand center: mean of all landmark (x, y)."""
    c = lm[:, :2].mean(axis=0)
    return (float(c[0]), float(c[1]))


def palm_normal(lm: np.ndarray) -> tuple[float, float, float]:
    """Unit normal of the palm plane, from (wrist->index MCP) x (wrist->pinky MCP)."""
    wrist = lm[WRIST]
    n = np.cross(lm[INDEX_MCP] - wrist, lm[PINKY_MCP] - wrist)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return (0.0, 0.0, 0.0)
    n = n / norm
    return (float(n[0]), float(n[1]), float(n[2]))


def palm_angle(lm: np.ndarray) -> float:
    """In-plane hand rotation: angle of the wrist -> middle-MCP vector, radians."""
    v = _xy(lm, MIDDLE_MCP) - _xy(lm, WRIST)
    return float(math.atan2(float(v[1]), float(v[0])))


class FeatureExtractor:
    """Turns a (smoothed) HandObservation into HandFeatures.

    Stateful only for velocity: it remembers the previous centroid and timestamp
    per hand (keyed by handedness) so it can differentiate position over time.
    """

    def __init__(self):
        self._prev = {}  # handedness -> (centroid_xy, timestamp)

    def extract(self, obs: HandObservation) -> HandFeatures:
        lm = np.asarray(obs.landmarks, dtype=float)
        c = centroid(lm)
        return HandFeatures(
            cursor_point=cursor_point(lm),
            pinch_distance=pinch_distance(lm),
            fingers_extended=fingers_extended(lm),
            centroid=c,
            velocity=self._velocity(obs.handedness, c, obs.timestamp),
            palm_normal=palm_normal(lm),
            palm_angle=palm_angle(lm),
            timestamp=obs.timestamp,
        )

    def _velocity(self, hand: str, c: tuple[float, float], ts: float) -> tuple[float, float]:
        prev = self._prev.get(hand)
        self._prev[hand] = (c, ts)
        if prev is None:
            return (0.0, 0.0)
        (px, py), pts = prev
        dt = ts - pts
        if dt <= 1e-6:
            return (0.0, 0.0)
        return ((c[0] - px) / dt, (c[1] - py) / dt)

    def reset(self) -> None:
        """Forget per-hand history (e.g. when tracking is lost)."""
        self._prev.clear()
