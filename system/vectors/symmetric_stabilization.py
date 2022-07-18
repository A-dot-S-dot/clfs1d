"""Provides assembling classes of symmetric stabilization vector.

The symmetric stabilization is defined via

    s(v, bi) = c*I((Dbi-gh(bi))*(Dv-gh(v)), Omega),

where c=h/2p and gh is a gradient approximation operator. It is more efficient
to calculate the above mentioned term differently. Obviously, we have

    s(v, bi) = c*((Dv-g(v), Dbi)-(Dv-g(v), g(bi))),

where (.,.) denotes the L2-product. Therefore, we divide the above mentioned
term in two parts

    (Dv-g(v), Dbi)     and     (Dv-g(v), g(bi)).

Using a gradient approximation matrix G_ij = g_j(bi), we obtain for the second
term

    (Dv-g(v), g(bi)) = G_ij*(Dv-g(v), bj),

where we use Einstein's sum convention. The adventage is obviously, that G_ij can
be calculated in preprocessing.

"""

import numpy as np
from fem import FiniteElementSpace
from system.matrices import SystemMatrix
from system.matrices.gradient_approximation import GradientApproximationMatrix

from .builder import DOFVectorBuilder
from .gradient_correction import (
    GradientCorrectionGradientEntryCalculator,
    GradientCorrectionBuilder,
    GradientCorrectionEntryCalculator,
)


class SymmetricStabilizationBuilder(DOFVectorBuilder):
    _stabilization_factor: float
    _gradient_approximation_builder: DOFVectorBuilder

    _gradient_correction_builder: GradientCorrectionBuilder
    _gradient_correction_gradient_builder: GradientCorrectionBuilder

    _gradient_approximation_matrix: SystemMatrix

    def __init__(
        self,
        element_space: FiniteElementSpace,
        stabilization_parameter: float,
        gradient_approximation_builder: DOFVectorBuilder,
    ):
        self._element_space = element_space
        self._build_stabilization_factor(stabilization_parameter)

        self._gradient_approximation_builder = gradient_approximation_builder
        self._build_gradient_correction_builder()
        self._build_gradient_correction_gradient_builder()

        self._build_gradient_approximation_matrix()

    def _build_stabilization_factor(self, stabilization_parameter: float):
        self._stabilization_factor = (
            stabilization_parameter
            * self._element_space.mesh.step_length
            / (2 * self._element_space.polynomial_degree)
        )

    def _build_gradient_correction_builder(self):
        entry_calculator = GradientCorrectionEntryCalculator(
            self._element_space,
            self._element_space.polynomial_degree + 1,
        )

        self._gradient_correction_builder = GradientCorrectionBuilder(
            self._element_space, entry_calculator
        )

    def _build_gradient_correction_gradient_builder(self):
        entry_calculator = GradientCorrectionGradientEntryCalculator(
            self._element_space,
            self._element_space.polynomial_degree,
        )

        self._gradient_correction_gradient_builder = GradientCorrectionBuilder(
            self._element_space,
            entry_calculator,
        )

    def _build_gradient_approximation_matrix(self):
        self._gradient_approximation_matrix = GradientApproximationMatrix(
            self._element_space, self._gradient_approximation_builder
        )

    def build_dof(self, discrete_solution_dof_vector: np.ndarray) -> np.ndarray:
        gradient_approximation_dof_vector = (
            self._gradient_approximation_builder.build_dof(discrete_solution_dof_vector)
        )
        gradient_correction_vector = self._gradient_correction_builder.build_dof(
            discrete_solution_dof_vector, gradient_approximation_dof_vector
        )
        gradient_correction_gradient_vector = (
            self._gradient_correction_gradient_builder.build_dof(
                discrete_solution_dof_vector, gradient_approximation_dof_vector
            )
        )

        return self._stabilization_factor * (
            gradient_correction_gradient_vector
            - self._gradient_approximation_matrix.dot(gradient_correction_vector)
        )
