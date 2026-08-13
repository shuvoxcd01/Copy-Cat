from abc import ABC, abstractmethod


class ObservationCapture(ABC):
   """Abstract base class for observation captures."""

   @abstractmethod
   def capture_observation(self):
      raise NotImplementedError("Subclasses must implement this method")