"""Provides averaged gradient approximation matrix

    G_ij = g_j(vi)

where g_j is the chosen gradient approximation, e.g. averaged gradient or
l2-projection.

"""
import numpy as np
from fem import FiniteElementSpace

from .system_matrix import SystemMatrix
from .assembler import MatrixAssembler
from system.vectors import DOFVectorBuilder


class GradientApproximationAssembler(MatrixAssembler):
    _element_space: FiniteElementSpace
    _gradient_approximation_builder: DOFVectorBuilder

    def __init__(
        self,
        element_space: FiniteElementSpace,
        gradient_approximation_builder: DOFVectorBuilder,
    ):
        self._element_space = element_space
        self._gradient_approximation_builder = gradient_approximation_builder

    def fill_entries(self, matrix):
        for global_index in range(self._element_space.dimension):
            basis_i_dof = self._build_basis_element_dof(global_index)
            gradient_approximation_dof = self._gradient_approximation_builder.build_dof(
                basis_i_dof
            )

            for i, gradient_approximation_dof_i in enumerate(
                gradient_approximation_dof
            ):
                if gradient_approximation_dof_i != 0:
                    matrix[global_index, i] = gradient_approximation_dof_i

    def _build_basis_element_dof(self, global_index: int) -> np.ndarray:
        basis_element = np.zeros(self._element_space.dimension)
        basis_element[global_index] = 1

        return basis_element


class GradientApproximationMatrix(SystemMatrix):
    """This system matrix contains the dofs of gradient approximation of each basis
    element of an element space."""

    def __init__(
        self,
        element_space: FiniteElementSpace,
        gradient_approximation_builder: DOFVectorBuilder,
    ):
        assembler = GradientApproximationAssembler(
            element_space, gradient_approximation_builder
        )

        SystemMatrix.__init__(self, element_space.dimension, assembler)
