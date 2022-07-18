"""This module provides continuous gradients for finite elements using L2 projection.

The main idea is to approximate the gradient of a finite element w with a finite
element function. The approximation itself is the L2 projection of the discrete
solution uh, i.e.

    g(w) = M^{-1} ((Dw, bi))_i,

where (.,.) denotes the L2-product and M the mass matrix.

See 'Entropy conservation property and entropy stabilization of high-order
continuous Galerkin approximations to scalar conservation laws', D. Kuzmin, M.
Quezada de Luna, 2020, p. 3

"""

import numpy as np
from fem import FiniteElementSpace
from fem.fast_element import QuadratureFastFiniteElement
from quadrature.local import LocalElementQuadrature
from system.matrices.mass import MassMatrix

from .builder import DOFVectorBuilder
from .discrete_l2_product import DiscreteL2ProductEntryCalculator


class GradientL2ProjectionEntryCalculator(DiscreteL2ProductEntryCalculator):
    _finite_element: QuadratureFastFiniteElement

    def __init__(self, element_space: FiniteElementSpace):
        local_quadrature = LocalElementQuadrature(element_space.polynomial_degree)
        DiscreteL2ProductEntryCalculator.__init__(self, element_space, local_quadrature)

        self._build_finite_element()

    def _build_finite_element(self):
        self._finite_element = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        self._finite_element.set_derivatives()

    def set_finite_element_dof(self, finite_element_dof: np.ndarray):
        self._finite_element.set_dof(finite_element_dof)

    def _left_function(self, simplex_index: int, quadrature_node_index: int) -> float:
        return self._finite_element.derivative(simplex_index, quadrature_node_index)


class GradientL2ProjectionBuilder(DOFVectorBuilder):
    _entry_calculator: GradientL2ProjectionEntryCalculator
    _mass: MassMatrix

    def __init__(self, element_space: FiniteElementSpace, mass: MassMatrix):
        entry_calculator = GradientL2ProjectionEntryCalculator(element_space)

        DOFVectorBuilder.__init__(self, element_space, entry_calculator)
        self._mass = mass

    def build_dof(self, discrete_solution_dof_vector: np.ndarray) -> np.ndarray:
        self._entry_calculator.set_finite_element_dof(discrete_solution_dof_vector)
        right_side = super().build_dof()

        return self._mass.inverse(right_side)
