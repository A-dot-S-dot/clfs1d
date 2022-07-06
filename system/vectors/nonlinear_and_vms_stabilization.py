"""Provides assembling classes of convex combination of a vms and
nonlinear (NVMS) stabilization vector.

The vms and nonlinear stabilization is defined via

    s(uh, vi) = c*I((1-ah)*Duh*Dvi + ah*(Duh-g(uh))*Dvi), Omega),

with

    ah = 1-min(omega, |Duh-gh|/|Duh|),

c=h/2p and gh is a gradient approximation.

"""

import numpy as np
from fem import FiniteElementSpace

from .builder import DOFVectorBuilder
from .l2_product_correction_basis import (
    L2ProductCorrectionBasisBuilder,
    L2ProductCorrectionDerivativeBasisEntryCalculator,
)


class L2ProductNVMSStabilizationDerivativeBasisEntryCalculator(
    L2ProductCorrectionDerivativeBasisEntryCalculator,
):
    """Calculate int_K (Duh - ah*g(uh))*vi, where vi denotes the FEM basis."""

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

        return (
            discrete_solution_derivative
            - self._alpha_h(discrete_solution_derivative, gradient_approximation_value)
            * gradient_approximation_value
        )

    def _alpha_h(
        self, discrete_solution_derivative: float, gradient_approximation_value: float
    ) -> float:
        if discrete_solution_derivative != 0:
            return 1 - np.minimum(
                self._stabilization_parameter,
                np.absolute(discrete_solution_derivative - gradient_approximation_value)
                / np.absolute(discrete_solution_derivative),
            )
        else:
            return 0


class NonlinearAndVMSStabilizationBuilder(DOFVectorBuilder):
    _element_space: FiniteElementSpace
    _stabilization_factor: float
    _gradient_approximation_builder: DOFVectorBuilder

    _l2_product_nvms_stabilization_derivative_basis_builder: L2ProductCorrectionBasisBuilder

    def __init__(
        self,
        element_space: FiniteElementSpace,
        stabilization_parameter: float,
        gradient_approximation_builder: DOFVectorBuilder,
    ):
        self._element_space = element_space
        self._build_stabilization_factor()

        self._gradient_approximation_builder = gradient_approximation_builder
        self._build_l2_product_nvms_stabilization_derivative_basis_builder(
            stabilization_parameter
        )

    def _build_stabilization_factor(self):
        self._stabilization_factor = self._element_space.mesh.step_length / (
            2 * self._element_space.polynomial_degree
        )

    def _build_l2_product_nvms_stabilization_derivative_basis_builder(
        self, stabilization_parameter: float
    ):
        l2_product_nvms_correction_derivative_basis_builder = (
            L2ProductNVMSStabilizationDerivativeBasisEntryCalculator(
                self._element_space,
                self._element_space.polynomial_degree,
                stabilization_parameter,
            )
        )

        self._l2_product_nvms_stabilization_derivative_basis_builder = (
            L2ProductCorrectionBasisBuilder(
                self._element_space,
                l2_product_nvms_correction_derivative_basis_builder,
            )
        )

    def build_vector(self, discrete_solution_dof_vector: np.ndarray) -> np.ndarray:
        gradient_approximation_dof_vector = (
            self._gradient_approximation_builder.build_vector(
                discrete_solution_dof_vector
            )
        )
        l2_product_nvms_stabilization_derivative_basis_vector = (
            self._l2_product_nvms_stabilization_derivative_basis_builder.build_vector(
                discrete_solution_dof_vector, gradient_approximation_dof_vector
            )
        )

        return self._stabilization_factor * (
            l2_product_nvms_stabilization_derivative_basis_vector
        )
