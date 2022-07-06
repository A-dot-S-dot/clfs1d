"""Provides assembling classes of symmetric stabilization vector.

The symmetric stabilization is defined via

    s(uh, vi) = c*I((Dvi-gh(vi))*(Duh-gh(uh)), Omega),

where c=h/2p and gh is a gradient approximation operator. It is more efficient
to calculate the above mentioned term differently. Obviously, we have

    s(uh, vi) = c*((Duh-g(uh), Dvi)-(Duh-g(uh), g(vi))),

where (.,.) denotes the L2-product. Therefore, we divide the above mentioned
term in two parts

    (Duh-g(uh), Dvi)     and     (Duh-g(uh), g(vi)).

Using a gradient approximation matrix G_ij = g_j(vi), we obtain for the second
term

    (Duh-g(uh), g(vi)) = G_ij*(Duh-g(uh), vj),

where we use Einstein's sum convention. The adventage is obviously, that G_ij can
be calculated in preprocessing.

"""

import numpy as np
from fem import FiniteElementSpace
from system.matrices import SystemMatrix
from system.matrices.gradient_approximation import GradientApproximationMatrix

from .builder import DOFVectorBuilder
from .l2_product_correction_basis import (
    L2ProductCorrectionBasisEntryCalculator,
    L2ProductCorrectionDerivativeBasisEntryCalculator,
    L2ProductCorrectionBasisBuilder,
)


class SymmetricStabilizationBuilder(DOFVectorBuilder):
    _element_space: FiniteElementSpace
    _stabilization_factor: float
    _gradient_approximation_builder: DOFVectorBuilder

    _l2_product_correction_basis_builder: L2ProductCorrectionBasisBuilder
    _l2_product_correction_derivative_basis_builder: L2ProductCorrectionBasisBuilder

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
        self._build_l2_product_correction_basis_builder()
        self._build_correction_derivative_builder()

        self._build_gradient_approximation_matrix()

    def _build_stabilization_factor(self, stabilization_parameter: float):
        self._stabilization_factor = (
            stabilization_parameter
            * self._element_space.mesh.step_length
            / (2 * self._element_space.polynomial_degree)
        )

    def _build_l2_product_correction_basis_builder(self):
        l2_product_correction_basis_entry_calculator = (
            L2ProductCorrectionBasisEntryCalculator(
                self._element_space,
                self._element_space.polynomial_degree + 1,
            )
        )

        self._l2_product_correction_basis_builder = L2ProductCorrectionBasisBuilder(
            self._element_space, l2_product_correction_basis_entry_calculator
        )

    def _build_correction_derivative_builder(self):
        l2_product_correction_derivative_basis_entry_calculator = (
            L2ProductCorrectionDerivativeBasisEntryCalculator(
                self._element_space,
                self._element_space.polynomial_degree,
            )
        )

        self._l2_product_correction_derivative_basis_builder = (
            L2ProductCorrectionBasisBuilder(
                self._element_space,
                l2_product_correction_derivative_basis_entry_calculator,
            )
        )

    def _build_gradient_approximation_matrix(self):
        self._gradient_approximation_matrix = GradientApproximationMatrix(
            self._element_space, self._gradient_approximation_builder
        )

    def build_vector(self, discrete_solution_dof_vector: np.ndarray) -> np.ndarray:
        gradient_approximation_dof_vector = (
            self._gradient_approximation_builder.build_vector(
                discrete_solution_dof_vector
            )
        )
        l2_product_correction_basis_vector = (
            self._l2_product_correction_basis_builder.build_vector(
                discrete_solution_dof_vector, gradient_approximation_dof_vector
            )
        )
        l2_product_correction_derivative_basis_vector = (
            self._l2_product_correction_derivative_basis_builder.build_vector(
                discrete_solution_dof_vector, gradient_approximation_dof_vector
            )
        )

        return self._stabilization_factor * (
            l2_product_correction_derivative_basis_vector
            - self._gradient_approximation_matrix.dot(
                l2_product_correction_basis_vector
            )
        )
