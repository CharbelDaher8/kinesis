"""Composition root: build concrete adapters, inject into Pipeline, run. Start with DryRunAdapter. The ONE place allowed to import from adapters/."""
from kinesis.adapters.dryrun import DryRunAdapter
from kinesis.adapters.mediapipe_tracker import MediaPipeTracker
from kinesis.adapters.webcam import WebcamSource
from kinesis.app.pipeline import Pipeline
from kinesis.domain.gestures.engine import GestureEngine
from kinesis.domain.gestures.pinch import PinchRecognizer
from kinesis.domain.gestures.point import PointRecognizer


def build() -> Pipeline:
    """Wire the concrete adapters into the pipeline. Swap DryRunAdapter for
    OSCursorAdapter here once the gestures feel right."""
    return Pipeline(
        source=WebcamSource(),
        tracker=MediaPipeTracker(),
        engine=GestureEngine([PinchRecognizer, PointRecognizer]),
        output=DryRunAdapter(),
    )


def main() -> None:
    print("kinesis — running (DryRun; nothing touches your real cursor). Ctrl-C to stop.")
    pipeline = build()
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nstopping.")
        pipeline.stop()


if __name__ == "__main__":
    main()
