"""Pipeline: per-frame loop source -> tracker -> smoothing -> features -> gestures -> mapper -> output. Depends on PORTS only; concrete adapters injected by main.py."""
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
        self._running = False

    def run(self) -> None:
        self._running = True
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

        hands = self.tracker.track(frame)
        context = self.context.current() if self.context else None

        features = [
            (obs.handedness, self.features.extract(self.smoother.smooth(obs)))
            for obs in hands
        ]

        # Engagement gate (global): present if any hand, engaged if any is posed.
        engage = any(self._engage_signal(feats) for _, feats in features)
        self.engagement.update(hand_present=bool(hands), engage=engage)

        for hand, feats in features:
            events = self.engine.update(feats, hand)  # advance recognizers regardless
            if not self.engagement.engaged:
                continue
            for event in events:
                intent = self.mapper.map(event, context=context)
                if intent is not None:
                    self.output.handle(intent)
        return True

    def _engage_signal(self, feats) -> bool:
        """Controlling posture: index finger up, or a near-pinch."""
        _thumb, index, *_rest = feats.fingers_extended
        return bool(index) or feats.pinch_distance < 0.5

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        for resource in (self.source, self.tracker, self.output):
            try:
                resource.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
