"""Provides assembling classes of convex combination of a nonlinear and
symmetric (NSym) stabilization vector.

The symmetric and nonlinear stabilization is defined via

    s(v, bi) = c*I((1-ah)*Dv*Dbi + ah*(Dv-g(v))*(Dbi-g(bi)), Omega),

with

    ah = 1-min(omega, |Dv-gh|/|Dv|),

c=h/2p and gh is a gradient approximation. It is more efficient to calculate the
above mentioned term differently. Obviously, we have

    s(v, bi) = c*((Dv-ah*g(v), Dbi)-(ah*(Dv-g(v)), g(bi))),

where (.,.) denotes the L2-product. Therefore, we divide the above mentioned
term in two parts

    (Dv-ah*g(v), Dbi)     and     (ah*(Dv-g(v)), g(bi)).

Using a gradient approximation matrix G_ij = g_j(bi), we obtain for the second
term

    (ah*(Dv-g(v)), g(bi)) = G_ij*(ah*(Dv-g(v)), bj),

where we use Einstein's sum convention. The adventage is obviously, that G_ij can
be calculated in preprocessing.

"""

import numpy as np
from fem import FiniteElementSpace
from system.matrices import SystemMatrix
from system.matrices.gradient_approximation import GradientApproximationMatrix

from .builder import DOFVectorBuilder
from .gradient_correction import (
    GradientCorrectionBuilder,
    GradientCorrectionEntryCalculator,
    GradientCorrectionGradientEntryCalculator,
)


class NSymEntryCalculator(GradientCorrectionEntryCalculator):
    """Calculate int_K ah*(Duh-g(uh))*bi, where bi denotes the FEM basis."""

    _stabilization_parameter: float

    def __init__(
        self,
        element_space: FiniteElementSpace,
        quadrature_degree: int,
        stabilization_parameter: float,
    ):
        GradientCorrectionEntryCalculator.__init__(
            self, element_space, quadrature_degree
        )
        self._stabilization_parameter = stabilization_parameter

    def _left_term(self, simplex_index: int, quadrature_node_index: int) -> float:
        finite_element_derivative = self._finite_element.derivative(
            simplex_index, quadrature_node_index
        )
        gradient_approximation_value = self._gradient_approximation.value(
            simplex_index, quadrature_node_index
        )

        return self._alpha_h(
            finite_element_derivative, gradient_approximation_value
        ) * (finite_element_derivative - gradient_approximation_value)

    def _alpha_h(
        self, finite_element_derivative: float, gradient_approximation_value: float
    ) -> float:
        if finite_element_derivative != 0:
            return 1 - np.minimum(
                self._stabilization_parameter,
                np.absolute(finite_element_derivative - gradient_approximation_value)
                / np.absolute(finite_element_derivative),
            )
        else:
            return 0


class NSymGradientEntryCalculator(
    GradientCorrectionGradientEntryCalculator,
    NSymEntryCalculator,
):
    """Calculate int_K (Duh - ah*g(uh))*bi, where bi denotes the FEM basis."""

    _stabilization_parameter: float

    def __init__(
        self,
        element_space: FiniteElementSpace,
        quadrature_degree: int,
        stabilization_parameter: float,
    ):
        GradientCorrectionGradientEntryCalculator.__init__(
            self, element_space, quadrature_degree
        )
        self._stabilization_parameter = stabilization_parameter

    def _left_term(self, simplex_index: int, quadrature_node_index: int) -> float:
        finite_element_derivative = self._finite_element.derivative(
            simplex_index, quadrature_node_index
        )
        gradient_approximation_value = self._gradient_approximation.value(
            simplex_index, quadrature_node_index
        )

        return (
            finite_element_derivative
            - self._alpha_h(finite_element_derivative, gradient_approximation_value)
            * gradient_approximation_value
        )


class NonlinearAndSymmetricStabilizationBuilder(DOFVectorBuilder):
    _element_space: FiniteElementSpace
    _stabilization_factor: float
    _gradient_approximation_builder: DOFVectorBuilder

    _nsym_builder: GradientCorrectionBuilder
    _nsym_gradient_builder: GradientCorrectionBuilder

    _gradient_approximation_matrix: SystemMatrix

    def __init__(
        self,
        element_space: FiniteElementSpace,
        stabilization_parameter: float,
        gradient_approximation_builder: DOFVectorBuilder,
    ):
        self._element_space = element_space
        self._build_stabilization_factor()

        self._gradient_approximation_builder = gradient_approximation_builder
        self._build_nsym_builder(stabilization_parameter)
        self._build_nsym_gradient_builder(stabilization_parameter)

        self._build_gradient_approximation_matrix()

    def _build_stabilization_factor(self):
        self._stabilization_factor = self._element_space.mesh.step_length / (
            2 * self._element_space.polynomial_degree
        )

    def _build_nsym_builder(self, stabilization_parameter: float):
        entry_calculator = NSymEntryCalculator(
            self._element_space,
            self._element_space.polynomial_degree + 1,
            stabilization_parameter,
        )

        self._nsym__builder = GradientCorrectionBuilder(
            self._element_space, entry_calculator
        )

    def _build_nsym_gradient_builder(self, stabilization_parameter: float):
        entry_calculator = NSymGradientEntryCalculator(
            self._element_space,
            self._element_space.polynomial_degree,
            stabilization_parameter,
        )

        self._nsym_gradient_builder = GradientCorrectionBuilder(
            self._element_space,
            entry_calculator,
        )

    def _build_gradient_approximation_matrix(self):
        self._gradient_approximation_matrix = GradientApproximationMatrix(
            self._element_space, self._gradient_approximation_builder
        )

    def build_dof(self, finite_element_dof: np.ndarray) -> np.ndarray:
        gradient_approximation_dof = self._gradient_approximation_builder.build_dof(
            finite_element_dof
        )
        nsym_discrete_l2_product = self._nsym__builder.build_dof(
            finite_element_dof, gradient_approximation_dof
        )
        nsym_discrete_gradient_l2_product = self._nsym_gradient_builder.build_dof(
            finite_element_dof, gradient_approximation_dof
        )

        return self._stabilization_factor * (
            nsym_discrete_gradient_l2_product
            - self._gradient_approximation_matrix.dot(nsym_discrete_l2_product)
        )
