"""One-Euro smoothing filter (pure math) — removes landmark jitter.

The One-Euro filter (Casiez, Roussel & Vogel, 2012) is an adaptive low-pass
filter: a low cutoff at rest kills jitter, and the cutoff rises with speed so
fast motion isn't laggy. It's the standard fix for shaky cursor control.

All operations are elementwise, so one filter smooths a whole (21, 3) landmark
array at once, with a per-coordinate adaptive cutoff.
"""
import math

import numpy as np

from kinesis.domain.types import HandObservation


def _alpha(cutoff, dt: float):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Adaptive low-pass filter for a scalar or numpy array signal.

    min_cutoff: cutoff at rest (lower = smoother but laggier).
    beta:       how fast the cutoff opens up with speed (higher = less lag).
    Tune both against a real recording.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=float)
        if self._x_prev is None:
            self._x_prev, self._dx_prev, self._t_prev = x, np.zeros_like(x), t
            return x
        dt = t - self._t_prev
        if dt <= 1e-6:
            return self._x_prev

        # low-pass the derivative
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # speed-adaptive cutoff, then low-pass the signal
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat

    def reset(self) -> None:
        self._x_prev = self._dx_prev = self._t_prev = None


class HandSmoother:
    """Applies a One-Euro filter to each hand's landmarks, keyed by handedness."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.7, d_cutoff: float = 1.0):
        self._cfg = (min_cutoff, beta, d_cutoff)
        self._filters = {}  # handedness -> OneEuroFilter

    def smooth(self, obs: HandObservation) -> HandObservation:
        f = self._filters.get(obs.handedness)
        if f is None:
            f = OneEuroFilter(*self._cfg)
            self._filters[obs.handedness] = f
        return HandObservation(
            landmarks=f(obs.landmarks, obs.timestamp),
            handedness=obs.handedness,
            timestamp=obs.timestamp,
        )

    def reset(self) -> None:
        self._filters.clear()
