"""Map GestureEvents to Intents using the active AppContext (passed in as a value, not fetched here). Pure.

This stage is why gestures and actions are separated: the same PINCH can mean
different things in different apps. Per-app behavior is a profile lookup (data),
not branching sprinkled through the code.
"""
from kinesis.domain.types import (
    GestureEvent,
    GesturePhase,
    GestureType,
    Intent,
    IntentType,
)


def _move_cursor(ev: GestureEvent) -> Intent:
    x, y = ev.data.get("point", (0.0, 0.0))
    return Intent(IntentType.MOVE_CURSOR, ev.timestamp, {"x": x, "y": y})


def _click(ev: GestureEvent) -> Intent:
    return Intent(IntentType.CLICK, ev.timestamp, {"button": "left"})


def _swipe(ev: GestureEvent) -> Intent:
    direction = ev.data.get("direction")
    kind = IntentType.PREV_PAGE if direction == "left" else IntentType.NEXT_PAGE
    return Intent(kind, ev.timestamp, {})


# A profile maps (gesture type, phase) -> a factory that builds the Intent.
DEFAULT_PROFILE = {
    (GestureType.POINT, GesturePhase.BEGAN): _move_cursor,
    (GestureType.POINT, GesturePhase.CHANGED): _move_cursor,
    (GestureType.PINCH, GesturePhase.BEGAN): _click,
    (GestureType.SWIPE, GesturePhase.RECOGNIZED): _swipe,
}


class IntentMapper:
    """Turns GestureEvents into semantic Intents for the active app context.

    Pure: the context (e.g. frontmost app name) is passed to map() as a value —
    this class never queries the OS. Register per-app profiles to override the
    default; unknown contexts fall back to the default.
    """

    def __init__(self, profiles: dict | None = None, default: dict = DEFAULT_PROFILE):
        self._default = default
        self._profiles = profiles or {}  # context -> profile

    def map(self, event: GestureEvent, context: str | None = None) -> Intent | None:
        profile = self._profiles.get(context, self._default)
        factory = profile.get((event.type, event.phase))
        return factory(event) if factory else None
