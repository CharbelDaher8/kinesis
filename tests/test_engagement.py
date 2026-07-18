"""Unit tests for the engagement FSM — pure transitions.

Run: PYTHONPATH=. .venv/bin/python tests/test_engagement.py
"""
from kinesis.domain.engagement import EngagementFSM, EngagementState


def _engaged():
    fsm = EngagementFSM()
    fsm.update(hand_present=True, engage=False)  # -> ARMED
    fsm.update(hand_present=True, engage=True)   # -> ENGAGED
    return fsm


def test_starts_idle():
    fsm = EngagementFSM()
    assert fsm.state is EngagementState.IDLE
    assert not fsm.engaged


def test_idle_to_armed_when_hand_appears():
    fsm = EngagementFSM()
    assert fsm.update(hand_present=True, engage=False) is EngagementState.ARMED


def test_armed_to_engaged_on_engage_signal():
    fsm = _engaged()
    assert fsm.state is EngagementState.ENGAGED
    assert fsm.engaged


def test_engaged_back_to_armed_when_disengaged():
    fsm = _engaged()
    assert fsm.update(hand_present=True, engage=False) is EngagementState.ARMED
    assert not fsm.engaged


def test_menu_opens_and_closes():
    fsm = _engaged()
    assert fsm.update(True, engage=True, menu=True) is EngagementState.MENU
    assert not fsm.engaged  # cursor paused while in menu
    assert fsm.update(True, engage=True, menu=False) is EngagementState.ENGAGED


def test_hand_lost_returns_to_idle_from_any_state():
    fsm = _engaged()
    assert fsm.update(hand_present=False, engage=True) is EngagementState.IDLE
    assert not fsm.engaged


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    _run()
