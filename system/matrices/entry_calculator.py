"""This module provides the interface and some implementations for objects which
calculate entries of system matrices."""
from abc import ABC, abstractmethod

import numpy as np
from fem import FiniteElementSpace, LocalFiniteElement
from fem.lagrange import LagrangeFiniteElementSpace, LocalLagrangeBasis
from quadrature.local import LocalElementQuadrature


class SystemMatrixEntryCalculator(ABC):
    """Object for calculating a matrix entry."""

    @abstractmethod
    def __call__(
        self, simplex_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        ...


class ConstantSystemMatrixEntryCalculator(SystemMatrixEntryCalculator):
    """Calculates the entries of the global system matrix by calculating a local one.

    In most cases this is used for uniform meshes and DOF independent matrices.

    """

    _local_quadrature: LocalElementQuadrature
    _local_basis: LocalLagrangeBasis
    _local_matrix: np.ndarray

    def __init__(self, element_space: FiniteElementSpace, quadrature_degree: int):
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)

        if not isinstance(element_space, LagrangeFiniteElementSpace):
            raise NotImplementedError

        self._local_basis = element_space.local_basis
        self._local_matrix = np.zeros(
            (element_space.indices_per_simplex, element_space.indices_per_simplex)
        )
        self._fill_local_matrix()

    def _fill_local_matrix(self):
        for local_index_1 in range(len(self._local_basis)):
            for local_index_2 in range(len(self._local_basis)):
                element_1 = self._local_basis[local_index_1]
                element_2 = self._local_basis[local_index_2]
                self._local_matrix[
                    local_index_1, local_index_2
                ] = self._fill_local_matrix_entry(element_1, element_2)

    def _fill_local_matrix_entry(
        self, element_1: LocalFiniteElement, element_2: LocalFiniteElement
    ) -> float:
        raise NotImplementedError(
            "`_fill_local_matrix_entry` must be implemented by subclasses"
        )

    def __call__(
        self, simplex_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        return self._local_matrix[local_index_1, local_index_2]
