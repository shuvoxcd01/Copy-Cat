from abc import ABC, abstractmethod

class ActionCapture(ABC):
    """Abstract base class for action captures."""

    @abstractmethod
    def capture_action(self):
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def noop_action(self):
        """The action value meaning nothing was done during a poll window."""
        return 0