"""Pipeline: per-frame loop source -> tracker -> smoothing -> features -> gestures -> mapper -> output. Depends on PORTS only; concrete adapters injected by main.py."""
import time

from kinesis.domain.engagement import EngagementFSM
from kinesis.domain.features import FeatureExtractor
from kinesis.domain.gestures.engine import GestureEngine
from kinesis.domain.intent.mapper import IntentMapper
from kinesis.domain.smoothing import HandSmoother
from kinesis.ports.context_provider import AppContextProvider
from kinesis.ports.frame_source import FrameSource
from kinesis.ports.hand_tracker import HandTracker
from kinesis.ports.output import OutputAdapter


class Pipeline:
    """Runs the per-frame loop and owns the domain stages.

    Depends on PORTS (source, tracker, output, context) — concrete adapters are
    injected by main.py, the composition root. The domain stages (smoothing,
    features, engine, mapper, engagement) have sensible defaults but can be
    swapped for tuning or testing.

    `grace` frames of latch keep engagement alive through a brief tracking blip
    (the pointing fist occludes fingers, so MediaPipe loses it for a few frames);
    without it every blip would drop control for a fraction of a second.
    """

    def __init__(
        self,
        source: FrameSource,
        tracker: HandTracker,
        engine: GestureEngine,
        output: OutputAdapter,
        *,
        smoother: HandSmoother | None = None,
        features: FeatureExtractor | None = None,
        mapper: IntentMapper | None = None,
        engagement: EngagementFSM | None = None,
        context: AppContextProvider | None = None,
        grace: int = 8,
        log_fps: bool = False,
    ):
        self.source = source
        self.tracker = tracker
        self.engine = engine
        self.output = output
        self.smoother = smoother or HandSmoother()
        self.features = features or FeatureExtractor()
        self.mapper = mapper or IntentMapper()
        self.engagement = engagement or EngagementFSM()
        self.context = context
        self.grace = grace
        self.log_fps = log_fps
        self._running = False
        self._since_hand = grace + 1  # start disengaged
        self._since_engage = grace + 1
        self._fps_count = 0
        self._fps_t0 = 0.0
        self._compute_sum = 0.0
        self._compute_max = 0.0

    def run(self) -> None:
        self._running = True
        self._fps_t0 = time.perf_counter()
        try:
            while self._running and self.step():
                pass
        finally:
            self.close()

    def step(self) -> bool:
        """Process one frame. Returns False when the source is exhausted."""
        frame = self.source.read()
        if frame is None:
            return False
        t0 = time.perf_counter() if self.log_fps else 0.0

        hands = self.tracker.track(frame)
        context = self.context.current() if self.context else None

        features = [
            (obs.handedness, self.features.extract(self.smoother.smooth(obs)))
            for obs in hands
        ]

        # Grace latch: hold "present"/"engaged" for a few frames after they drop,
        # so a momentary tracking or pose blip doesn't disengage control.
        raw_engage = any(self._engage_signal(feats) for _, feats in features)
        self._since_hand = 0 if hands else self._since_hand + 1
        self._since_engage = 0 if raw_engage else self._since_engage + 1
        hand_present = self._since_hand <= self.grace
        engage = self._since_engage <= self.grace
        self.engagement.update(hand_present=hand_present, engage=engage)

        for hand, feats in features:
            events = self.engine.update(feats, hand)  # advance recognizers regardless
            if not self.engagement.engaged:
                continue
            for event in events:
                intent = self.mapper.map(event, context=context)
                if intent is not None:
                    self.output.handle(intent)

        if self.log_fps:
            dt = time.perf_counter() - t0
            self._compute_sum += dt
            self._compute_max = max(self._compute_max, dt)
        self._tick_fps()
        return True

    def _engage_signal(self, feats) -> bool:
        """Controlling posture: index finger up, or a near-pinch."""
        _thumb, index, *_rest = feats.fingers_extended
        return bool(index) or feats.pinch_distance < 0.5

    def _tick_fps(self) -> None:
        if not self.log_fps:
            return
        self._fps_count += 1
        elapsed = time.perf_counter() - self._fps_t0
        if elapsed >= 2.0:
            avg_ms = 1000.0 * self._compute_sum / max(self._fps_count, 1)
            print(f"[fps] {self._fps_count / elapsed:4.1f}   compute avg {avg_ms:4.1f}ms  max {1000.0 * self._compute_max:5.1f}ms")
            self._fps_count, self._fps_t0 = 0, time.perf_counter()
            self._compute_sum, self._compute_max = 0.0, 0.0

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        for resource in (self.source, self.tracker, self.output):
            try:
                resource.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
