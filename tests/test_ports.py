"""Ports are abstract interfaces: unusable directly, implementable by adapters.

Run: PYTHONPATH=. .venv/bin/python tests/test_ports.py
"""
from kinesis.domain.types import Intent, IntentType
from kinesis.ports.context_provider import AppContextProvider
from kinesis.ports.frame_source import FrameSource
from kinesis.ports.hand_tracker import HandTracker
from kinesis.ports.output import OutputAdapter


def _assert_abstract(cls):
    try:
        cls()
    except TypeError:
        return
    raise AssertionError(f"{cls.__name__} should be abstract")


def test_ports_are_abstract():
    for cls in (FrameSource, HandTracker, OutputAdapter, AppContextProvider):
        _assert_abstract(cls)


def test_ports_can_be_implemented():
    class FS(FrameSource):
        def read(self):
            return None

    class HT(HandTracker):
        def track(self, frame):
            return []

    class Out(OutputAdapter):
        def __init__(self):
            self.seen = []

        def handle(self, intent):
            self.seen.append(intent)

    class Ctx(AppContextProvider):
        def current(self):
            return "Test"

    assert FS().read() is None
    assert HT().track(None) == []
    out = Out()
    out.handle(Intent(IntentType.CLICK, 0.0))
    assert len(out.seen) == 1
    assert Ctx().current() == "Test"


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _run()
