"""End-to-end wiring test: scripted source + tracker through the real domain
stack into a DryRun adapter. No camera, no MediaPipe.

Run: PYTHONPATH=. .venv/bin/python tests/test_pipeline.py
"""
import numpy as np

from kinesis.adapters.dryrun import DryRunAdapter
from kinesis.app.pipeline import Pipeline
from kinesis.domain.gestures.engine import GestureEngine
from kinesis.domain.gestures.pinch import PinchRecognizer
from kinesis.domain.gestures.point import PointRecognizer
from kinesis.domain.types import Frame, HandObservation, IntentType
from kinesis.ports.frame_source import FrameSource
from kinesis.ports.hand_tracker import HandTracker

# Base open hand (all fingers extended).
_BASE = np.array([
    [0.50, 0.90, 0.0], [0.42, 0.82, 0.0], [0.37, 0.74, 0.0], [0.33, 0.68, 0.0],
    [0.30, 0.63, 0.0], [0.45, 0.60, 0.0], [0.44, 0.48, 0.0], [0.43, 0.40, 0.0],
    [0.43, 0.33, 0.0], [0.52, 0.58, 0.0], [0.52, 0.45, 0.0], [0.52, 0.36, 0.0],
    [0.52, 0.28, 0.0], [0.58, 0.60, 0.0], [0.59, 0.47, 0.0], [0.59, 0.39, 0.0],
    [0.59, 0.32, 0.0], [0.64, 0.64, 0.0], [0.66, 0.54, 0.0], [0.67, 0.47, 0.0],
    [0.68, 0.42, 0.0],
])


def _pointing():
    """Fist with index out: middle/ring/pinky curled toward the palm."""
    lm = _BASE.copy()
    lm[12] = [0.52, 0.62, 0.0]
    lm[16] = [0.58, 0.62, 0.0]
    lm[20] = [0.63, 0.64, 0.0]
    return lm


def _open_pinch():
    """Open hand (fingers extended) with the thumb tip brought onto the index tip."""
    lm = _BASE.copy()
    lm[4] = [0.43, 0.35, 0.0]
    return lm


class _NoSmoother:
    def smooth(self, obs):
        return obs

    def reset(self):
        pass


class ScriptedSource(FrameSource):
    def __init__(self, n):
        self.n, self.i = n, 0

    def read(self):
        if self.i >= self.n:
            return None
        frame = Frame(image=None, timestamp=self.i / 30.0, frame_id=self.i)
        self.i += 1
        return frame

    def close(self):
        pass


class ScriptedTracker(HandTracker):
    def __init__(self, arrays):
        self.arrays, self.i = arrays, 0

    def track(self, frame):
        arr = self.arrays[min(self.i, len(self.arrays) - 1)]
        self.i += 1
        return [HandObservation(landmarks=arr, handedness="Right", timestamp=frame.timestamp)]

    def close(self):
        pass


def test_pipeline_turns_gestures_into_intents():
    # point a few frames (cursor moves), then hold an open-hand pinch (debounce -> click)
    seq = [_pointing()] * 3 + [_open_pinch()] * 5
    out = DryRunAdapter(printer=lambda _s: None)
    Pipeline(
        source=ScriptedSource(len(seq)),
        tracker=ScriptedTracker(seq),
        engine=GestureEngine([PinchRecognizer, PointRecognizer]),
        output=out,
        smoother=_NoSmoother(),  # isolate wiring from smoothing dynamics
    ).run()

    kinds = [i.type for i in out.history]
    assert IntentType.MOVE_CURSOR in kinds  # from the pointing pose
    assert IntentType.CLICK in kinds        # from the open-hand pinch


def test_pipeline_ignores_fist_pinch():
    # a tight thumb-index gap but in the fist/point pose must NOT click
    lm = _pointing()
    lm[4] = [0.43, 0.35, 0.0]  # thumb tucked onto index while middle/ring/pinky curled
    out = DryRunAdapter(printer=lambda _s: None)
    Pipeline(
        source=ScriptedSource(6),
        tracker=ScriptedTracker([lm]),
        engine=GestureEngine([PinchRecognizer, PointRecognizer]),
        output=out,
        smoother=_NoSmoother(),
    ).run()
    assert IntentType.CLICK not in [i.type for i in out.history]


def test_pipeline_stays_idle_with_no_hands():
    class Empty(HandTracker):
        def track(self, frame):
            return []

        def close(self):
            pass

    out = DryRunAdapter(printer=lambda _s: None)
    Pipeline(
        source=ScriptedSource(3),
        tracker=Empty(),
        engine=GestureEngine([PinchRecognizer, PointRecognizer]),
        output=out,
        smoother=_NoSmoother(),
    ).run()
    assert out.history == []


class GappyTracker(HandTracker):
    """Returns a hand per frame, or [] where the scripted array is None (a blip)."""

    def __init__(self, arrays):
        self.arrays, self.i = arrays, 0

    def track(self, frame):
        arr = self.arrays[self.i]
        self.i += 1
        return [] if arr is None else [HandObservation(arr, "Right", frame.timestamp)]

    def close(self):
        pass


def _pipe_over(seq, grace):
    return Pipeline(
        source=ScriptedSource(len(seq)),
        tracker=GappyTracker(seq),
        engine=GestureEngine([PinchRecognizer, PointRecognizer]),
        output=DryRunAdapter(printer=lambda _s: None),
        smoother=_NoSmoother(),
        grace=grace,
    )


def test_grace_holds_engagement_through_a_blip():
    seq = [_pointing(), _pointing(), _pointing(), None, _pointing()]  # frame 3 = tracking blip
    pipe = _pipe_over(seq, grace=8)
    pipe.step()
    pipe.step()  # -> ENGAGED
    assert pipe.engagement.engaged
    pipe.step()
    pipe.step()  # the blip frame
    assert pipe.engagement.engaged  # grace kept control alive


def test_no_grace_disengages_on_a_blip():
    seq = [_pointing(), _pointing(), _pointing(), None, _pointing()]
    pipe = _pipe_over(seq, grace=0)
    pipe.step()
    pipe.step()
    pipe.step()
    assert pipe.engagement.engaged
    pipe.step()  # blip with no grace
    assert not pipe.engagement.engaged


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _run()
