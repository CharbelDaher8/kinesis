"""Composition root: build concrete adapters, inject into Pipeline, run. Start with DryRunAdapter. The ONE place allowed to import from adapters/."""
import sys

from kinesis.adapters.dryrun import DryRunAdapter
from kinesis.adapters.mediapipe_tracker import MediaPipeTracker
from kinesis.adapters.os_cursor import OSCursorAdapter
from kinesis.adapters.webcam import WebcamSource
from kinesis.app.pipeline import Pipeline
from kinesis.domain.gestures.engine import GestureEngine
from kinesis.domain.gestures.pinch import PinchRecognizer
from kinesis.domain.gestures.point import PointRecognizer


def build(live: bool = False) -> Pipeline:
    """Wire the concrete adapters into the pipeline. `live` swaps the safe DryRun
    adapter for the real cursor."""
    output = OSCursorAdapter() if live else DryRunAdapter()
    return Pipeline(
        source=WebcamSource(),
        tracker=MediaPipeTracker(),
        engine=GestureEngine([PinchRecognizer, PointRecognizer]),
        output=output,
        log_fps=True,
    )


def _install_killswitch(pipeline: Pipeline) -> None:
    """Global ESC to hard-stop — a safety net if the cursor gets away from you."""
    from pynput import keyboard

    def on_press(key):
        if key == keyboard.Key.esc:
            pipeline.stop()
            return False  # stop the listener

    keyboard.Listener(on_press=on_press).start()


def main() -> None:
    live = "--live" in sys.argv
    pipeline = build(live=live)
    if live:
        print("LIVE — controlling your real cursor. Drop your hand to disengage; ESC or Ctrl-C to quit.")
        _install_killswitch(pipeline)
    else:
        print("DryRun — printing intents, not touching the cursor. Ctrl-C to quit.  (pass --live for real control)")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
