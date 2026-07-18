"""Unit tests for the real-cursor adapter — a fake mouse, so nothing actually moves.

Run: PYTHONPATH=. .venv/bin/python tests/test_os_cursor.py
"""
from pynput.mouse import Button

from kinesis.adapters.os_cursor import OSCursorAdapter
from kinesis.domain.types import Intent, IntentType


class FakeMouse:
    def __init__(self):
        self.position = (0.0, 0.0)
        self.clicks = []
        self.scrolls = []

    def click(self, button, count):
        self.clicks.append((button, count))

    def scroll(self, dx, dy):
        self.scrolls.append((dx, dy))


def _adapter(mouse):
    # smooth=False -> moves synchronously, so assertions are deterministic (no thread)
    return OSCursorAdapter(screen_size=(1000.0, 800.0), mouse=mouse, smooth=False)


def test_move_maps_normalized_to_pixels():
    m = FakeMouse()
    _adapter(m).handle(Intent(IntentType.MOVE_CURSOR, 0.0, {"x": 0.5, "y": 0.25}))
    assert m.position == (500.0, 200.0)


def test_move_clamps_out_of_range():
    m = FakeMouse()
    _adapter(m).handle(Intent(IntentType.MOVE_CURSOR, 0.0, {"x": 1.5, "y": -0.2}))
    assert m.position == (1000.0, 0.0)


def test_click_presses_left():
    m = FakeMouse()
    _adapter(m).handle(Intent(IntentType.CLICK, 0.0, {"button": "left"}))
    assert m.clicks == [(Button.left, 1)]


def test_scroll_forwards_deltas():
    m = FakeMouse()
    _adapter(m).handle(Intent(IntentType.SCROLL, 0.0, {"dx": 0, "dy": -3}))
    assert m.scrolls == [(0, -3)]


def test_smooth_mode_eases_toward_target():
    import time

    m = FakeMouse()
    a = OSCursorAdapter(screen_size=(1000.0, 800.0), mouse=m, smooth=True, responsiveness=0.6, rate_hz=240)
    try:
        a.handle(Intent(IntentType.MOVE_CURSOR, 0.0, {"x": 0.5, "y": 0.5}))
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            if abs(m.position[0] - 500.0) < 1.0 and abs(m.position[1] - 400.0) < 1.0:
                break
            time.sleep(0.01)
        assert abs(m.position[0] - 500.0) < 2.0  # converged toward the target
        assert abs(m.position[1] - 400.0) < 2.0
    finally:
        a.close()


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _run()
