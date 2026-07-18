"""AppContextProvider port: return the current app context (frontmost app). Implemented by adapters/macos_context.py.

The interface only. The mapper consumes the returned value; the OS-specific
lookup (NSWorkspace, pyobjc) lives in the adapter.
"""
from abc import ABC, abstractmethod


class AppContextProvider(ABC):
    @abstractmethod
    def current(self) -> str | None:
        """Name of the frontmost application, or None if unknown."""
