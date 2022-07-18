"""Provides a classes which produces a vector containing L2 products with basis elements or its derivatives."""

from typing import List

import numpy as np
from fem import (
    FastMapping,
    FastLocalElement,
    FiniteElementSpace,
    QuadratureFastFiniteElement,
)
from fem.lagrange import LagrangeFiniteElementSpace, LocalLagrangeBasis
from quadrature.local import LocalElementQuadrature

from .builder import DOFVectorBuilder
from .entry_calculator import DOFEntryCalculator


class DiscreteL2ProductEntryCalculator(DOFEntryCalculator):
    """This class is made for building the L2 product between a onedimensional function f
    and a finite element basis (bi), i.e. the built vector is

    vi = (f,bi).

    """

    _element_space: FiniteElementSpace
    _local_quadrature: LocalElementQuadrature
    _local_basis_elements: List[FastLocalElement]
    _determinant_derivative_affine_mapping: float
    _left_function_object: FastMapping

    def __init__(
        self,
        element_space: FiniteElementSpace,
        local_quadrature: LocalElementQuadrature,
    ):
        if not isinstance(element_space, LagrangeFiniteElementSpace):
            raise NotImplementedError

        self._element_space = element_space
        self._local_quadrature = local_quadrature
        self._build_local_basis_elements()
        self._determinant_derivative_affine_mapping = element_space.mesh.step_length

    def _build_local_basis_elements(self):
        fast_element = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        fast_element.set_values()

        self._local_basis_elements = fast_element.local_basis_elements

    def set_left_function(self, function: FastMapping):
        self._left_function_object = function

    def __call__(self, simplex_index: int, local_index: int) -> float:
        right_function = self._local_basis_elements[local_index]

        node_values = np.array(
            [
                self._left_function(simplex_index, node_index)
                * right_function.value(node_index)
                for node_index in range(len(self._local_quadrature.nodes))
            ]
        )

        return self._determinant_derivative_affine_mapping * np.dot(
            self._local_quadrature.weights, node_values
        )

    def _left_function(self, simplex_index: int, quadrature_node_index: int) -> float:
        return self._left_function_object.value(simplex_index, quadrature_node_index)


class DiscreteGradientL2ProductEntryCalculator(DiscreteL2ProductEntryCalculator):
    """This class is made for building the L2 product between a onedimensional function f
    and the derivatives of a finite element basis (bi), i.e. the built vector is

    vi = (f,bi').

    """

    _local_basis: LocalLagrangeBasis

    def __init__(
        self,
        element_space: FiniteElementSpace,
        local_quadrature: LocalElementQuadrature,
    ):
        DiscreteL2ProductEntryCalculator.__init__(self, element_space, local_quadrature)

    def _build_local_basis_elements(self):
        fast_element = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        fast_element.set_derivatives()

        self._local_basis_elements = fast_element.local_basis_elements

    def __call__(self, simplex_index: int, local_index: int) -> float:
        right_function = self._local_basis_elements[local_index]

        node_values = np.array(
            [
                self._left_function(simplex_index, node_index)
                * right_function.derivative(node_index)
                for node_index in range(len(self._local_quadrature.nodes))
            ]
        )

        # The transformation determinant and the derivative of affine
        # transformation cancel each other out

        return np.dot(self._local_quadrature.weights, node_values)


class DiscreteL2ProductBuilder(DOFVectorBuilder):
    _entry_calculator: DiscreteL2ProductEntryCalculator

    def __init__(
        self, element_space: FiniteElementSpace, quadrature: LocalElementQuadrature
    ):
        self._entry_calculator = DiscreteL2ProductEntryCalculator(
            element_space, quadrature
        )
        DOFVectorBuilder.__init__(self, element_space, self._entry_calculator)

    def set_left_function(self, function: FastMapping):
        self._entry_calculator.set_left_function(function)


class DiscreteGradientL2ProductBuilder(DOFVectorBuilder):
    _entry_calculator: DiscreteGradientL2ProductEntryCalculator

    def __init__(
        self, element_space: FiniteElementSpace, quadrature: LocalElementQuadrature
    ):
        self._entry_calculator = DiscreteGradientL2ProductEntryCalculator(
            element_space, quadrature
        )
        DOFVectorBuilder.__init__(self, element_space, self._entry_calculator)

    def set_left_function(self, function: FastMapping):
        self._entry_calculator.set_left_function(function)
