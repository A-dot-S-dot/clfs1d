from typing import List

import numpy as np
from math_types import BarycentricCoordinate, FunctionRealToReal
from mesh import Simplex

from .abstracts import Quadrature
from .gauss import GaussianQuadratureGeneralized


class LocalElementQuadrature(Quadrature):
    _domain = Simplex(0, 1)
    _nodes: List[BarycentricCoordinate]
    _weights: np.ndarray
    _local_coordinates = _domain.local_coordinates
    _world_coordinates = _domain.world_coordinates

    def __init__(self, nodes_number: int):
        """The quadrature is 2*NODES_NUMBER-1 exact."""
        gauss_quadrature = GaussianQuadratureGeneralized(nodes_number, self._domain)

        self._build_nodes(gauss_quadrature.nodes)
        self._weights = gauss_quadrature.weights

    def _build_nodes(self, nodes: List[float]):
        self._nodes = [self._local_coordinates(node) for node in nodes]

    @property
    def nodes(self) -> List[BarycentricCoordinate]:
        return self._nodes

    def integrate(self, function: FunctionRealToReal) -> float:
        quadrature_nodes_values = np.array(
            [function(self._world_coordinates(node)) for node in self.nodes]
        )

        return np.dot(self.weights, quadrature_nodes_values)
