"""OSCursorAdapter(OutputAdapter): real cursor/scroll via pynput/Quartz. Needs macOS Accessibility permission. Avoid pyautogui default 0.1s PAUSE."""
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
    pixels, so the same intents work on any display. `mouse` and `screen_size` are
    injectable for testing without moving the real cursor.
    """

    def __init__(self, screen_size: tuple[float, float] | None = None, mouse=None):
        self._mouse = mouse or Controller()
        self._w, self._h = screen_size or _main_screen_size()

    def handle(self, intent: Intent) -> None:
        if intent.type is IntentType.MOVE_CURSOR:
            x = _clamp01(intent.params.get("x", 0.0))
            y = _clamp01(intent.params.get("y", 0.0))
            self._mouse.position = (x * self._w, y * self._h)
        elif intent.type is IntentType.CLICK:
            self._mouse.click(Button.left, 1)
        elif intent.type is IntentType.SCROLL:
            self._mouse.scroll(intent.params.get("dx", 0), intent.params.get("dy", 0))
        # ZOOM / NEXT_PAGE / PREV_PAGE arrive later (keyboard shortcuts).

    def close(self) -> None:
        pass
