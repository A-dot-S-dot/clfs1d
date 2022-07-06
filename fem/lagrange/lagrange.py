"""Provides Lagrange Finite Elements."""

import numpy as np
from math_types import FunctionRealToReal
from mesh import Interval, UniformMesh

from ..abstracts import FiniteElementSpace
from ..dof_index_mapping import DOFIndexMapping, PeriodicDOFIndexMapping
from .local_lagrange import LocalLagrangeBasis


class LagrangeFiniteElementSpace(FiniteElementSpace):
    _mesh: UniformMesh
    _local_basis: LocalLagrangeBasis
    _dof_index_mapping: DOFIndexMapping
    _polynomial_degree: int

    def __init__(self, mesh: UniformMesh, polynomial_degree: int):
        self._mesh = mesh
        self._polynomial_degree = polynomial_degree
        self._local_basis = LocalLagrangeBasis(polynomial_degree)
        self._build_index_mapping()

    def _build_index_mapping(self):
        self._dof_index_mapping = PeriodicDOFIndexMapping(
            self._mesh, len(self._local_basis)
        )

    @property
    def polynomial_degree(self) -> int:
        return self._polynomial_degree

    @property
    def dimension(self) -> int:
        return self._dof_index_mapping.output_dimension

    @property
    def domain(self) -> Interval:
        return self._mesh.domain

    @property
    def mesh(self) -> UniformMesh:
        return self._mesh

    @property
    def indices_per_simplex(self):
        return self.polynomial_degree + 1

    @property
    def local_basis(self):
        return self._local_basis

    def get_global_index(self, simplex_index: int, local_index: int) -> int:
        return self._dof_index_mapping(simplex_index, local_index)

    def get_value(self, point: float, dof_vector: np.ndarray) -> float:
        simplex_index = self._mesh.find_simplex_index(point)[0]
        return self.get_value_on_simplex(point, dof_vector, simplex_index)

    def get_value_on_simplex(
        self, point: float, dof_vector: np.ndarray, simplex_index: int
    ) -> float:
        simplex = self._mesh[simplex_index]
        value = 0

        for local_index, local_element in zip(
            range(len(self._local_basis)), self._local_basis
        ):
            global_index = self._dof_index_mapping(simplex_index, local_index)
            value += dof_vector[global_index] * local_element(
                simplex.local_coordinates(point)
            )

        return value

    def get_derivative(self, point: float, dof_vector: np.ndarray) -> float:
        simplex_index = self._mesh.find_simplex_index(point)[0]
        simplex = self._mesh[simplex_index]

        if simplex.is_in_boundary(point):
            return np.nan
        else:
            return self.get_derivative_on_simplex(point, dof_vector, simplex_index)

    def get_derivative_on_simplex(
        self, point: float, dof_vector: np.ndarray, simplex_index: int
    ) -> float:
        simplex = self._mesh[simplex_index]

        value = 0
        for local_index, local_element in zip(
            range(len(self._local_basis)), self._local_basis
        ):
            global_index = self._dof_index_mapping(simplex_index, local_index)
            local_derivative = np.array(
                local_element.derivative(simplex.local_coordinates(point))
            )
            value += dof_vector[global_index] * local_derivative

        return np.dot(simplex.Lambda, value)

    def interpolate(self, f: FunctionRealToReal) -> np.ndarray:
        dof_vector = self._empty_dof()

        for simplex, simplex_index in zip(self._mesh, range(len(self._mesh))):
            for local_index, node in zip(
                range(len(self._local_basis)), self._local_basis.nodes
            ):
                global_index = self._dof_index_mapping(simplex_index, local_index)
                point = simplex.world_coordinates(node)
                f_point = f(point)

                if np.isnan(dof_vector[global_index]):
                    dof_vector[global_index] = f_point
                elif dof_vector[global_index] != f_point:
                    raise ValueError(f"the given function is not continuous in {point}")

        return dof_vector

    def _empty_dof(self) -> np.ndarray:
        empty_dof_vector = np.zeros(self.dimension)
        empty_dof_vector[:] = np.nan

        return empty_dof_vector
