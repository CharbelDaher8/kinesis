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


def _flag_float(flag: str, default: float) -> float:
    """Read `--flag VALUE` from argv, or return the default."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return float(sys.argv[i + 1])
    return default


def build(live: bool = False, sensitivity: float = 2.0) -> Pipeline:
    """Wire the concrete adapters into the pipeline. `live` swaps the safe DryRun
    adapter for the real cursor; `sensitivity` is the cursor gain (DPI)."""
    output = OSCursorAdapter(sensitivity=sensitivity) if live else DryRunAdapter()
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
    sensitivity = _flag_float("--sensitivity", 2.0)
    pipeline = build(live=live, sensitivity=sensitivity)
    if live:
        print(f"LIVE (sensitivity {sensitivity:g}) — controlling your real cursor. "
              f"Drop your hand to disengage; ESC or Ctrl-C to quit.")
        _install_killswitch(pipeline)
    else:
        print("DryRun — printing intents, not touching the cursor. Ctrl-C to quit.  (pass --live for real control)")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
