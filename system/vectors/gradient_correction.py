"""Provides objects for building the vectors

    ((Dw-g(w))*phi_i)_i and ((Dw-g(w))*Dphi_i)_i

where (.,.) denotes the L2 product, w a finite element and g is gradient approximation.

"""
from typing import Union

import numpy as np
from fem import FiniteElementSpace
from fem.fast_element import QuadratureFastFiniteElement
from quadrature.local import LocalElementQuadrature

from .builder import DOFVectorBuilder
from .entry_calculator import DOFEntryCalculator
from .discrete_l2_product import (
    DiscreteL2ProductEntryCalculator,
    DiscreteGradientL2ProductEntryCalculator,
)


class GradientCorrectionEntryCalculator(DiscreteL2ProductEntryCalculator):
    """This class is made for building the L2 product between Dw-g(w)
    and a finite element basis (bi), i.e. the built vector is

    vi = (Dw-g(w),bi),

    where w is a finite element and g a gradient approximation.

    """

    _finite_element: QuadratureFastFiniteElement
    _gradient_approximation: QuadratureFastFiniteElement

    def __init__(self, element_space: FiniteElementSpace, quadrature_degree: int):
        local_quadrature = LocalElementQuadrature(quadrature_degree)
        DiscreteL2ProductEntryCalculator.__init__(self, element_space, local_quadrature)

        self._build_gradient_approximation()
        self._build_finite_element()

    def _build_gradient_approximation(self):
        self._gradient_approximation = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        self._gradient_approximation.set_values()

    def _build_finite_element(self):
        self._finite_element = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        self._finite_element.set_derivatives()

    def set_finite_element_dof(
        self,
        discrete_solution_dof: np.ndarray,
    ):
        self._finite_element.set_dof(discrete_solution_dof)

    def set_gradient_approximation_dof(self, gradient_approximation_dof: np.ndarray):
        self._gradient_approximation.set_dof(gradient_approximation_dof)

    def _left_function(self, simplex_index: int, quadrature_node_index: int) -> float:
        finite_element_derivative = self._finite_element.derivative(
            simplex_index, quadrature_node_index
        )
        gradient_approximation_value = self._gradient_approximation.value(
            simplex_index, quadrature_node_index
        )

        return finite_element_derivative - gradient_approximation_value


class GradientCorrectionGradientEntryCalculator(
    DiscreteGradientL2ProductEntryCalculator,
    GradientCorrectionEntryCalculator,
):
    """This class is made for building the L2 product between Dw-g(w) and the
    derivatives of a finite element basis (bi), i.e. the built vector is

    vi = (Dw-g(w),Dbi),

    where w is a finite element and g a gradient approximation.

    """

    def __init__(self, element_space: FiniteElementSpace, quadrature_degree: int):
        local_quadrature = LocalElementQuadrature(quadrature_degree)
        DiscreteGradientL2ProductEntryCalculator.__init__(
            self, element_space, local_quadrature
        )

        self._build_gradient_approximation()
        self._build_finite_element()


class GradientCorrectionBuilder(DOFVectorBuilder):
    _entry_calculator: GradientCorrectionEntryCalculator

    def __init__(
        self,
        element_space: FiniteElementSpace,
        entry_calculator: GradientCorrectionEntryCalculator,
    ):
        DOFVectorBuilder.__init__(self, element_space, entry_calculator)

    def build_dof(
        self,
        discrete_solution_dof_vector: np.ndarray,
        gradient_approximation_dof_vector: np.ndarray,
    ) -> np.ndarray:
        self._entry_calculator.set_finite_element_dof(discrete_solution_dof_vector)
        self._entry_calculator.set_gradient_approximation_dof(
            gradient_approximation_dof_vector
        )

        return DOFVectorBuilder.build_dof(self)
