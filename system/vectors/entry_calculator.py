from abc import ABC, abstractmethod


class DOFVectorEntryCalculator(ABC):
    """Class for calculating a vector entry using local to global principles."""

    @abstractmethod
    def __call__(self, simplex_index: int, local_index: int) -> float:
        ...
