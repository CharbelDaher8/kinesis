"""Unit tests for domain.smoothing — deterministic, no camera.

Run: PYTHONPATH=. .venv/bin/python tests/test_smoothing.py
"""
import statistics

import numpy as np

from kinesis.domain.smoothing import HandSmoother, OneEuroFilter
from kinesis.domain.types import HandObservation

DT = 1.0 / 30.0


def test_first_sample_passes_through():
    f = OneEuroFilter()
    out = f(np.array([0.42, 0.10]), 0.0)
    assert list(out) == [0.42, 0.10]


def test_reduces_jitter_on_a_still_signal():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    rng = np.random.default_rng(0)
    noisy_res, filt_res = [], []
    for i in range(300):
        noisy = 0.5 + rng.normal(0.0, 0.05)
        out = float(f(np.array([noisy]), i * DT)[0])
        noisy_res.append(noisy - 0.5)
        filt_res.append(out - 0.5)
    # after warmup, the filtered signal is markedly steadier
    assert statistics.pstdev(filt_res[50:]) < 0.6 * statistics.pstdev(noisy_res[50:])


def test_tracks_a_moving_signal_without_large_lag():
    f = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    last = 0.0
    for i in range(200):
        last = float(f(np.array([0.001 * i]), i * DT)[0])
    assert abs(last - 0.001 * 199) < 0.02  # keeps up with the ramp


def test_hand_smoother_keeps_shape_and_actually_smooths():
    s = HandSmoother()
    lm = np.random.default_rng(1).random((21, 3))
    o1 = s.smooth(HandObservation(lm, "Right", 0.0))
    assert o1.landmarks.shape == (21, 3)
    o2 = s.smooth(HandObservation(lm + 0.1, "Right", DT))
    assert o2.landmarks.shape == (21, 3)
    assert not np.allclose(o2.landmarks, lm + 0.1)  # pulled toward the previous frame


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _run()
