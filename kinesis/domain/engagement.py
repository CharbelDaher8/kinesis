"""Engagement supervisor FSM: IDLE -> ARMED -> ENGAGED -> MENU. Pure policy; gates whether intents are acted on.

Nothing downstream acts unless this says ENGAGED. That gate is what stops stray
hand motion (reaching for coffee) from firing clicks, and it's how you disengage
by simply dropping your hand.

Pure: it is fed derived boolean signals each frame and returns the new state. The
caller decides what 'engage'/'menu' mean from features (open palm, pinch-hold,
...), so this policy stays decoupled from geometry.
"""
from enum import Enum


class EngagementState(Enum):
    IDLE = "idle"        # no hand tracked
    ARMED = "armed"      # hand present, not actively controlling
    ENGAGED = "engaged"  # actively controlling — the only state that emits intents
    MENU = "menu"        # radial menu open; cursor control paused


class EngagementFSM:
    def __init__(self):
        self.state = EngagementState.IDLE

    def update(self, hand_present: bool, engage: bool, menu: bool = False) -> EngagementState:
        s = self.state
        if not hand_present:
            s = EngagementState.IDLE
        elif s is EngagementState.IDLE:
            s = EngagementState.ARMED
        elif s is EngagementState.ARMED:
            if engage:
                s = EngagementState.ENGAGED
        elif s is EngagementState.ENGAGED:
            if menu:
                s = EngagementState.MENU
            elif not engage:
                s = EngagementState.ARMED
        elif s is EngagementState.MENU:
            if not menu:
                s = EngagementState.ENGAGED
        self.state = s
        return s

    @property
    def engaged(self) -> bool:
        """True only when gestures should drive actions."""
        return self.state is EngagementState.ENGAGED

    def reset(self) -> None:
        self.state = EngagementState.IDLE
