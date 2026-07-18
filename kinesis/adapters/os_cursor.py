"""OSCursorAdapter(OutputAdapter): real cursor/scroll via pynput/Quartz. Needs macOS Accessibility permission. Avoid pyautogui default 0.1s PAUSE."""
import threading
import time

from AppKit import NSScreen
from pynput.mouse import Button, Controller

from kinesis.domain.types import Intent, IntentType
from kinesis.ports.output import OutputAdapter


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _main_screen_size() -> tuple[float, float]:
    size = NSScreen.mainScreen().frame().size
    return (float(size.width), float(size.height))


class OSCursorAdapter(OutputAdapter):
    """Drives the real macOS cursor via pynput. Requires Accessibility permission.

    Intent params are normalized (0..1); this is the one place they become screen
    pixels, so the same intents work on any display.

    With `smooth=True` (default), MOVE_CURSOR only updates a *target*, and a
    background loop eases the real cursor toward it at `rate_hz`, so 30fps tracking
    still renders as glassy 120Hz cursor motion. `responsiveness` (0..1) trades lag
    for smoothness: higher = snappier, lower = smoother. `mouse`/`screen_size` are
    injectable, and `smooth=False` moves synchronously (used by tests).
    """

    def __init__(
        self,
        screen_size: tuple[float, float] | None = None,
        mouse=None,
        smooth: bool = True,
        responsiveness: float = 0.5,
        rate_hz: int = 120,
    ):
        self._mouse = mouse or Controller()
        self._w, self._h = screen_size or _main_screen_size()
        self._smooth = smooth
        self._alpha = responsiveness
        self._target: tuple[float, float] | None = None
        self._current: tuple[float, float] | None = None
        self._lock = threading.Lock()
        self._running = smooth
        self._thread = None
        if smooth:
            self._thread = threading.Thread(target=self._render_loop, args=(rate_hz,), daemon=True)
            self._thread.start()

    def handle(self, intent: Intent) -> None:
        if intent.type is IntentType.MOVE_CURSOR:
            px = _clamp01(intent.params.get("x", 0.0)) * self._w
            py = _clamp01(intent.params.get("y", 0.0)) * self._h
            if self._smooth:
                with self._lock:
                    self._target = (px, py)
            else:
                self._mouse.position = (px, py)
        elif intent.type is IntentType.CLICK:
            self._mouse.click(Button.left, 1)
        elif intent.type is IntentType.SCROLL:
            self._mouse.scroll(intent.params.get("dx", 0), intent.params.get("dy", 0))
        # ZOOM / NEXT_PAGE / PREV_PAGE arrive later (keyboard shortcuts).

    def _render_loop(self, rate_hz: int) -> None:
        dt = 1.0 / rate_hz
        while self._running:
            with self._lock:
                target = self._target
            if target is not None:
                cur = self._current or target
                nx = cur[0] + (target[0] - cur[0]) * self._alpha
                ny = cur[1] + (target[1] - cur[1]) * self._alpha
                self._current = (nx, ny)
                self._mouse.position = (nx, ny)
            time.sleep(dt)

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
