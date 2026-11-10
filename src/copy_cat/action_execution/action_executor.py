from abc import ABC, abstractmethod


class ActionExecutor(ABC):
    """Abstract base class for action executors, the inverse of an ActionCapture.

    An executor turns the discrete actions a policy chooses back into input, using the
    same numbering the matching capture records with.
    """

    @abstractmethod
    def apply(self, action: int) -> None:
        """Make the input reflect an action."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def release(self) -> None:
        """Undo whatever input is still in effect, leaving things as they were found."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def noop_action(self) -> int:
        """The action value meaning do nothing."""
        return 0

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, releasing any input still in effect."""
        self.release()
