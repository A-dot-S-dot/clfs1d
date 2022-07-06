from typing import List

import numpy as np
from math_types import BarycentricCoordinate
from quadrature.local import LocalElementQuadrature

from .abstracts import FiniteElementSpace, LocalFiniteElementBasis


class FastLagrangeElement:
    _dof_vector: np.ndarray
    _current_simplex_index: int
    _local_dof_vector: np.ndarray

    _element_space: FiniteElementSpace
    _local_basis: LocalFiniteElementBasis
    _Lambda: np.ndarray

    _values: List[np.ndarray]
    _derivatives: List[np.ndarray]
    _local_value_points: List[BarycentricCoordinate]
    _local_derivative_points: List[BarycentricCoordinate]

    def __init__(self, element_space: FiniteElementSpace):
        self._element_space = element_space
        self._local_basis = element_space.local_basis
        self._local_value_points = []
        self._local_derivative_points = []
        self._values = []
        self._derivatives = []

        self._build_Lambda()

    def _build_Lambda(self):
        step_length = self._element_space.mesh.step_length
        self._Lambda = np.array([-1 / step_length, 1 / step_length])

    def set_dof_vector(self, dof_vector: np.ndarray):
        self._dof_vector = dof_vector
        self._current_simplex_index = -1

    @property
    def local_value_points(self) -> List[BarycentricCoordinate]:
        return self._local_value_points

    @property
    def local_derivative_points(self) -> List[BarycentricCoordinate]:
        return self._local_derivative_points

    def add_values(self, *local_points: BarycentricCoordinate):
        for local_point in local_points:
            self._local_value_points.append(local_point)
            self._values.append(
                np.array(
                    [basis_element(local_point) for basis_element in self._local_basis]
                )
            )

    def add_derivatives(self, *local_points: BarycentricCoordinate):
        for local_point in local_points:
            self._local_derivative_points.append(local_point)
            self._derivatives.append(
                np.array(
                    [
                        np.dot(self._Lambda, basis_element.derivative(local_point))
                        for basis_element in self._local_basis
                    ]
                )
            )

    def value_on_simplex(self, simplex_index: int, local_point_index: int) -> float:
        self._set_current_simplex(simplex_index)
        return np.dot(self._local_dof_vector, self._values[local_point_index])

    def _set_current_simplex(self, simplex_index: int):
        if simplex_index != self._current_simplex_index:
            self._current_simplex_index = simplex_index
            self._build_local_dofs(simplex_index)

    def _build_local_dofs(self, simplex_index: int):
        self._local_dof_vector = np.zeros(self._element_space.indices_per_simplex)
        for local_index in range(self._element_space.indices_per_simplex):
            global_index = self._element_space.get_global_index(
                simplex_index, local_index
            )
            self._local_dof_vector[local_index] = self._dof_vector[global_index]

    def derivative_on_simplex(
        self, simplex_index: int, local_point_index: int
    ) -> float:
        self._set_current_simplex(simplex_index)
        return np.dot(self._local_dof_vector, self._derivatives[local_point_index])


class QuadratureFastFiniteElement(FastLagrangeElement):
    _quadrature_nodes: List[BarycentricCoordinate]

    def __init__(
        self,
        element_space: FiniteElementSpace,
        local_quadrature: LocalElementQuadrature,
    ):
        FastLagrangeElement.__init__(self, element_space)

        self._quadrature_nodes = local_quadrature.nodes

    def add_values(self):
        super().add_values(*self._quadrature_nodes)

    def add_derivatives(self):
        super().add_derivatives(*self._quadrature_nodes)


class AnchorNodesFastFiniteElement(FastLagrangeElement):
    _anchor_nodes: List[BarycentricCoordinate]

    def __init__(self, element_space: FiniteElementSpace):
        FastLagrangeElement.__init__(self, element_space)

        self._anchor_nodes = self._local_basis.nodes

    def add_values(self):
        super().add_values(*self._anchor_nodes)

    def add_derivatives(self):
        super().add_derivatives(*self._anchor_nodes)
