"""OutputAdapter port: consume an Intent and perform it. Implemented by adapters/dryrun.py, os_cursor.py, overlay.py.

This is the intent -> effect seam. Swapping DryRun for OSCursor for Overlay is
just choosing a different implementation of this port — no change to the core.
"""
from abc import ABC, abstractmethod

from kinesis.domain.types import Intent


class OutputAdapter(ABC):
    @abstractmethod
    def handle(self, intent: Intent) -> None:
        """Perform (or log) a single intent."""

    def close(self) -> None:
        """Flush / release resources. Override if needed."""
