from typing import Iterator, List, Sequence

import numpy as np

from .interval import Interval


class UniformMesh:
    _step_length: float
    _domain: Interval
    _grid: np.ndarray
    _simplices: Sequence[Interval]

    def __init__(self, domain: Interval, elements_number: int) -> None:
        if elements_number <= 0:
            raise ValueError("{elements_number} is not positive")

        self._domain = domain

        nodes_number = elements_number + 1
        self._grid, step_length = np.linspace(
            domain.a, domain.b, nodes_number, retstep=True
        )

        self._step_length = float(step_length)

        self._build_simplices()

    def _build_simplices(self):
        self._simplices = []
        for index in range(len(self)):
            self._simplices.append(Interval(self._grid[index], self._grid[index + 1]))

    @property
    def step_length(self) -> float:
        return self._step_length

    @property
    def domain(self) -> Interval:
        return self._domain

    def __iter__(self) -> Iterator[Interval]:
        return iter(self._simplices)

    def __len__(self) -> int:
        return len(self._grid) - 1

    def __getitem__(self, index: int) -> Interval:
        return self._simplices[index]

    def __eq__(self, other) -> bool:
        return self.step_length == other.step_length and self.domain == other.domain

    def index(self, simplex: Interval) -> int:
        return self._simplices.index(simplex)

    def find_simplex_index(self, point: float) -> List[int]:
        indices = []
        for simplex in self._simplices:
            if point in simplex:
                indices.append(self.index(simplex))

        return indices

    def find_simplex(self, point: float) -> List[Interval]:
        return [self[index] for index in self.find_simplex_index(point)]

    def refine(self):
        new_elements_number = 2 * len(self)

        return UniformMesh(self.domain, new_elements_number)
