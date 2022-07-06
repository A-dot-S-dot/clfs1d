from typing import Iterator, List, Tuple

import numpy as np
from math_types import (
    BarycentricCoordinate,
    FunctionBarycentricToReal,
    FunctionBarycentricToRealD,
)
from scipy.interpolate import lagrange

from ..abstracts import LocalFiniteElement, LocalFiniteElementBasis


class LocalLagrangeBasis(LocalFiniteElementBasis):
    """Basis of finite elements on the Gibbs simplex. Each basis elements can be
    identified with a node.

    """

    _polynomial_degree: int
    _basis_elements: List[LocalFiniteElement]
    _nodes: List[BarycentricCoordinate]

    def __init__(self, polynomial_degree: int):
        self._polynomial_degree = polynomial_degree
        self._build_nodes()
        self._build_elements()

    def _build_nodes(self):
        N = self.polynomial_degree
        self._nodes = [((N - i) / N, i / N) for i in range(N + 1)]

    def _build_elements(self):
        self._basis_elements = []
        N = self.polynomial_degree
        node_first_entry_values = np.array([(N - i) / N for i in range(N + 1)])

        for i in range(len(self.nodes)):

            self._basis_elements.append(
                self._create_local_finite_element(i, node_first_entry_values)
            )

    def _create_local_finite_element(
        self, index: int, node_first_entry_values: np.ndarray
    ) -> LocalFiniteElement:
        unit_vector = self._build_unit_vector(index)

        interpolation, interpolation_derivative = self._build_interpolation(
            node_first_entry_values, unit_vector
        )

        return LocalFiniteElement(interpolation, interpolation_derivative)

    def _build_unit_vector(self, index: int) -> np.ndarray:
        unit_vector = np.zeros(len(self.nodes))
        unit_vector[index] = 1

        return unit_vector

    def _build_interpolation(
        self, x_values: np.ndarray, y_values: np.ndarray
    ) -> Tuple[FunctionBarycentricToReal, FunctionBarycentricToRealD]:
        interpolation = lagrange(x_values, y_values)
        interpolation_derivative = interpolation.deriv()

        return (
            lambda point: interpolation(point[0]),
            lambda point: np.array([interpolation_derivative(point[0]), 0]),
        )

    def __len__(self) -> int:
        return len(self._basis_elements)

    @property
    def nodes(self) -> List[BarycentricCoordinate]:
        return self._nodes

    @property
    def polynomial_degree(self) -> int:
        return self._polynomial_degree

    def __iter__(self) -> Iterator[LocalFiniteElement]:
        return iter(self._basis_elements)

    def __getitem__(self, node_index: int) -> LocalFiniteElement:
        return self._basis_elements[node_index]

    def get_element_at_node(self, node) -> LocalFiniteElement:
        return self._basis_elements[self._nodes.index(node)]
