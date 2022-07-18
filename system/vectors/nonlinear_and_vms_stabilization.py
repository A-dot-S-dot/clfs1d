"""Provides assembling classes of convex combination of a vms and
nonlinear (NVMS) stabilization vector.

The vms and nonlinear stabilization is defined via

    s(v, bi) = c*I((1-ah)*Dv*Dbi + ah*(Dv-g(v))*Dbi), Omega),

with

    ah = 1-min(omega, |Dv-gh|/|Dv|),

c=h/2p and gh is a gradient approximation.

"""

import numpy as np
from fem import FiniteElementSpace

from .builder import DOFVectorBuilder
from .gradient_correction import (
    GradientCorrectionBuilder,
    GradientCorrectionEntryCalculator,
)


class NVMSEntryCalculator(
    GradientCorrectionEntryCalculator,
):
    """Calculate int_K (Duh - ah*g(uh))*bi, where bi denotes the FEM basis."""

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

        return (
            finite_element_derivative
            - self._alpha_h(finite_element_derivative, gradient_approximation_value)
            * gradient_approximation_value
        )

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


class NonlinearAndVMSStabilizationBuilder(DOFVectorBuilder):
    _element_space: FiniteElementSpace
    _stabilization_factor: float
    _gradient_approximation_builder: DOFVectorBuilder
    _nvms_builder: GradientCorrectionBuilder

    def __init__(
        self,
        element_space: FiniteElementSpace,
        stabilization_parameter: float,
        gradient_approximation_builder: DOFVectorBuilder,
    ):
        self._element_space = element_space
        self._build_stabilization_factor()

        self._gradient_approximation_builder = gradient_approximation_builder
        self._build_nvms_builder(stabilization_parameter)

    def _build_stabilization_factor(self):
        self._stabilization_factor = self._element_space.mesh.step_length / (
            2 * self._element_space.polynomial_degree
        )

    def _build_nvms_builder(self, stabilization_parameter: float):
        nvms_entry_calculator = NVMSEntryCalculator(
            self._element_space,
            self._element_space.polynomial_degree,
            stabilization_parameter,
        )

        self._nvms_builder = GradientCorrectionBuilder(
            self._element_space,
            nvms_entry_calculator,
        )

    def build_dof(self, finite_element_dof: np.ndarray) -> np.ndarray:
        gradient_approximation_dof = self._gradient_approximation_builder.build_dof(
            finite_element_dof
        )
        nvms_vector = self._nvms_builder.build_dof(
            finite_element_dof, gradient_approximation_dof
        )

        return self._stabilization_factor * (nvms_vector)
