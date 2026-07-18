"""DryRunAdapter(OutputAdapter): logs intents, no side effects. Build the whole stack against this before touching the real cursor."""
from kinesis.domain.types import Intent
from kinesis.ports.output import OutputAdapter


class DryRunAdapter(OutputAdapter):
    """Prints intents instead of performing them — the safe development default.

    Also keeps a `history` list so tests and debugging can inspect exactly what
    would have happened, without moving the real cursor. Swap this for
    OSCursorAdapter only once a gesture is solid.
    """

    def __init__(self, record: bool = True, printer=print):
        self.record = record
        self.history: list[Intent] = []
        self._printer = printer

    def handle(self, intent: Intent) -> None:
        if self.record:
            self.history.append(intent)
        self._printer(f"[dryrun] {intent.type.value:<12} {intent.params}")

    def close(self) -> None:
        pass
