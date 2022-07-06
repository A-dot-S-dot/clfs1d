"""Provides assembling classes of nonlinear stabilization vector.

The nonlinear stabilization is defined via

    s(uh, vi) = c*I(ah*Dvi*Duh, Omega),

with

    ah = min(omega, |Duh-gh|/|Duh|),

c=h/2p and gh is a gradient approximation. For stability reasons we use

    ah = min(omega*|Duh|, |Duh-gh|)

and

    s(uh, vi) = c*(ah*sgn(Duh), Dvi),

where (.,.) denotes the L2-product.

"""

import numpy as np
from fem import FiniteElementSpace

from .builder import DOFVectorBuilder
from .l2_product_correction_basis import (
    L2ProductCorrectionDerivativeBasisEntryCalculator,
    L2ProductCorrectionBasisBuilder,
)


class L2ProductNonlinearStabilizationDerivativeBasisEntryCalculator(
    L2ProductCorrectionDerivativeBasisEntryCalculator
):
    _stabilization_parameter: float

    def __init__(
        self,
        element_space: FiniteElementSpace,
        quadrature_degree: int,
        stabilization_parameter: float,
    ):
        L2ProductCorrectionDerivativeBasisEntryCalculator.__init__(
            self, element_space, quadrature_degree
        )
        self._stabilization_parameter = stabilization_parameter

    def _left_term(self, simplex_index: int, quadrature_node_index: int) -> float:
        discrete_solution_derivative = self._discrete_solution.derivative_on_simplex(
            simplex_index, quadrature_node_index
        )
        gradient_approximation_value = self._gradient_approximation.value_on_simplex(
            simplex_index, quadrature_node_index
        )

        return np.sign(discrete_solution_derivative) * np.minimum(
            self._stabilization_parameter * np.absolute(discrete_solution_derivative),
            np.absolute(discrete_solution_derivative - gradient_approximation_value),
        )


class NonlinearStabilizationBuilder(DOFVectorBuilder):
    _element_space: FiniteElementSpace
    _stabilization_factor: float
    _gradient_approximation_builder: DOFVectorBuilder
    _l2_product_correction_derivative_builder: L2ProductCorrectionBasisBuilder

    def __init__(
        self,
        element_space: FiniteElementSpace,
        stabilization_parameter: float,
        gradient_approximation_builder: DOFVectorBuilder,
    ):
        self._element_space = element_space
        self._build_stabilization_factor()
        self._gradient_approximation_builder = gradient_approximation_builder
        self._build_l2_product_builder(stabilization_parameter)

    def _build_stabilization_factor(self):
        self._stabilization_factor = self._element_space.mesh.step_length / (
            2 * self._element_space.polynomial_degree
        )

    def _build_l2_product_builder(self, stabilization_parameter: float):
        entry_calculator = (
            L2ProductNonlinearStabilizationDerivativeBasisEntryCalculator(
                self._element_space,
                self._element_space.polynomial_degree + 1,
                stabilization_parameter,
            )
        )
        self._l2_product_correction_derivative_builder = (
            L2ProductCorrectionBasisBuilder(self._element_space, entry_calculator)
        )

    def build_vector(self, discrete_solution_dof_vector: np.ndarray) -> np.ndarray:
        gradient_approximation_dof_vector = (
            self._gradient_approximation_builder.build_vector(
                discrete_solution_dof_vector
            )
        )
        return (
            self._stabilization_factor
            * self._l2_product_correction_derivative_builder.build_vector(
                discrete_solution_dof_vector, gradient_approximation_dof_vector
            )
        )
