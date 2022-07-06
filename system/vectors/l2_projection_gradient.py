"""This module provides continuous gradients for finite elements using L2 projection.

The main idea is to approximate the gradient of a discrete solution uh with a
finite element function. The approximation itself is the L2 projection of the
discrete solution uh, i.e.

    g(uh) = M^{-1} ((Duh, vi))_i,

where (.,.) denotes the L2-product and M the mass matrix.

See 'Entropy conservation property and entropy stabilization of high-order
continuous Galerkin approximations to scalar conservation laws', D. Kuzmin, M.
Quezada de Luna, 2020, p. 3

"""

import numpy as np
from fem import FiniteElementSpace
from fem.fast_element import QuadratureFastFiniteElement
from system.matrices.mass import MassMatrix
from quadrature.local import LocalElementQuadrature

from .builder import DOFVectorBuilder
from .l2_product_function_basis import (
    L2ProductBasisBuilder,
    L2ProductFunctionBasisEntryCalculator,
)


class L2ProjectionGradientRightSideEntryCalculator(
    L2ProductFunctionBasisEntryCalculator
):
    _element_space: FiniteElementSpace
    _mass: MassMatrix
    _discrete_solution: QuadratureFastFiniteElement
    _l2_product_builder: L2ProductBasisBuilder

    def __init__(self, element_space: FiniteElementSpace):
        local_quadrature = LocalElementQuadrature(element_space.polynomial_degree)
        L2ProductFunctionBasisEntryCalculator.__init__(
            self, element_space, local_quadrature
        )

        self._build_fast_discrete_solution()

    def _build_fast_discrete_solution(self):
        self._discrete_solution = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        self._discrete_solution.add_derivatives()

    def set_discrete_solution_dof_vector(self, dof_vector: np.ndarray):
        self._discrete_solution.set_dof_vector(dof_vector)
        self._build_left_functions_node_values()

    def _build_left_functions_node_values(self):
        self._left_function_nodes_values_per_simplex = []

        for simplex_index in range(len(self._element_space.mesh)):
            self._left_function_nodes_values_per_simplex.append(
                [
                    self._discrete_solution.derivative_on_simplex(
                        simplex_index, node_index
                    )
                    for node_index in range(self._local_quadrature.degree)
                ]
            )


class L2ProjectionGradientBuilder(DOFVectorBuilder):
    _entry_calculator: L2ProjectionGradientRightSideEntryCalculator
    _mass: MassMatrix

    def __init__(self, element_space: FiniteElementSpace, mass: MassMatrix):
        entry_calculator = L2ProjectionGradientRightSideEntryCalculator(element_space)

        DOFVectorBuilder.__init__(self, element_space, entry_calculator)
        self._mass = mass

    def build_vector(self, discrete_solution_dof_vector: np.ndarray) -> np.ndarray:
        self._entry_calculator.set_discrete_solution_dof_vector(
            discrete_solution_dof_vector
        )
        right_side = super().build_vector()

        return self._mass.inverse(right_side)
