from abc import ABC, abstractmethod

from fem import GlobalFiniteElement


class HyperbolicPDESolver(ABC):
    @property
    @abstractmethod
    def discrete_solution(self) -> GlobalFiniteElement:
        ...

    @abstractmethod
    def solve(self, target_time: float, time_steps_number: int):
        ...
