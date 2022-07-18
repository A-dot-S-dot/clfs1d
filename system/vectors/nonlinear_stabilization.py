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
from .gradient_correction import (
    GradientCorrectionBuilder,
    GradientCorrectionGradientEntryCalculator,
)


class NonlinearStabilizationEntryCalculator(GradientCorrectionGradientEntryCalculator):
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

    def _left_function(self, simplex_index: int, quadrature_node_index: int) -> float:
        finite_element_derivative = self._finite_element.derivative(
            simplex_index, quadrature_node_index
        )
        gradient_approximation_value = self._gradient_approximation.value(
            simplex_index, quadrature_node_index
        )

        return np.sign(finite_element_derivative) * np.minimum(
            self._stabilization_parameter * np.absolute(finite_element_derivative),
            np.absolute(finite_element_derivative - gradient_approximation_value),
        )


class NonlinearStabilizationBuilder(DOFVectorBuilder):
    _element_space: FiniteElementSpace
    _stabilization_factor: float
    _gradient_approximation_builder: DOFVectorBuilder
    _nonlinear_stabilization_builder: GradientCorrectionBuilder

    def __init__(
        self,
        element_space: FiniteElementSpace,
        stabilization_parameter: float,
        gradient_approximation_builder: DOFVectorBuilder,
    ):
        self._element_space = element_space
        self._build_stabilization_factor()
        self._gradient_approximation_builder = gradient_approximation_builder
        self._build_nonlinear_stabilization_builder(stabilization_parameter)

    def _build_stabilization_factor(self):
        self._stabilization_factor = self._element_space.mesh.step_length / (
            2 * self._element_space.polynomial_degree
        )

    def _build_nonlinear_stabilization_builder(self, stabilization_parameter: float):
        entry_calculator = NonlinearStabilizationEntryCalculator(
            self._element_space,
            self._element_space.polynomial_degree + 1,
            stabilization_parameter,
        )
        self._nonlinear_stabilization_builder = GradientCorrectionBuilder(
            self._element_space, entry_calculator
        )

    def build_dof(self, finite_element_dof: np.ndarray) -> np.ndarray:
        gradient_approximation_dof = self._gradient_approximation_builder.build_dof(
            finite_element_dof
        )
        return (
            self._stabilization_factor
            * self._nonlinear_stabilization_builder.build_dof(
                finite_element_dof, gradient_approximation_dof
            )
        )
