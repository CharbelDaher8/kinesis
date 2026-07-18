"""Unit tests for the dry-run output adapter.

Run: PYTHONPATH=. .venv/bin/python tests/test_dryrun.py
"""
from kinesis.adapters.dryrun import DryRunAdapter
from kinesis.domain.types import Intent, IntentType


def test_records_and_prints_intents():
    out = []
    a = DryRunAdapter(printer=out.append)
    a.handle(Intent(IntentType.CLICK, 0.0, {"button": "left"}))
    a.handle(Intent(IntentType.MOVE_CURSOR, 1.0, {"x": 0.5, "y": 0.5}))
    assert [i.type for i in a.history] == [IntentType.CLICK, IntentType.MOVE_CURSOR]
    assert len(out) == 2
    assert "click" in out[0]


def test_recording_can_be_disabled():
    a = DryRunAdapter(record=False, printer=lambda _s: None)
    a.handle(Intent(IntentType.CLICK, 0.0))
    assert a.history == []


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _run()
