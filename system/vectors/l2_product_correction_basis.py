"""Provides objects for building the vector:
((Du-g(u))*phi_i)_i and ((Du-g(u))*Dphi_i)_i
"""
from typing import Union

import numpy as np
from fem import FiniteElementSpace
from fem.fast_element import QuadratureFastFiniteElement
from quadrature.local import LocalElementQuadrature

from .builder import DOFVectorBuilder
from .entry_calculator import DOFVectorEntryCalculator
from .l2_product_function_basis import (
    L2ProductFunctionBasisEntryCalculator,
    L2ProductFunctionDerivativeBasisEntryCalculator,
)


class L2ProductCorrectionBasisEntryCalculator(L2ProductFunctionBasisEntryCalculator):
    """Calculate int_K (Du_h-g(u_h))*v_i, where v_i denotes the FEM basis."""

    _element_space: FiniteElementSpace
    _local_quadrature: LocalElementQuadrature

    _discrete_solution: QuadratureFastFiniteElement
    _gradient_approximation: QuadratureFastFiniteElement

    def __init__(self, element_space: FiniteElementSpace, quadrature_degree: int):
        local_quadrature = LocalElementQuadrature(quadrature_degree)
        L2ProductFunctionBasisEntryCalculator.__init__(
            self, element_space, local_quadrature
        )

        self._build_fast_gradient_approximation()
        self._build_fast_discrete_solution()

    def _build_fast_gradient_approximation(self):
        self._gradient_approximation = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        self._gradient_approximation.add_values()

    def _build_fast_discrete_solution(self):
        self._discrete_solution = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        self._discrete_solution.add_derivatives()

    def set_discrete_solution_dof_vector(
        self,
        discrete_solution_dof_vector: np.ndarray,
    ):
        self._discrete_solution.set_dof_vector(discrete_solution_dof_vector)

    def set_gradient_approximation_dof_vector(
        self, gradient_approximation_dof_vector: np.ndarray
    ):
        self._gradient_approximation.set_dof_vector(gradient_approximation_dof_vector)

    def build_left_functions_node_values(self):
        self._left_function_nodes_values_per_simplex = []

        for simplex_index in range(len(self._element_space.mesh)):
            self._left_function_nodes_values_per_simplex.append(
                [
                    self._left_term(simplex_index, node_index)
                    for node_index in range(self._local_quadrature.degree)
                ]
            )

    def _left_term(self, simplex_index: int, quadrature_node_index: int) -> float:
        discrete_solution_derivative = self._discrete_solution.derivative_on_simplex(
            simplex_index, quadrature_node_index
        )
        gradient_approximation_value = self._gradient_approximation.value_on_simplex(
            simplex_index, quadrature_node_index
        )

        return discrete_solution_derivative - gradient_approximation_value


class L2ProductCorrectionDerivativeBasisEntryCalculator(
    L2ProductFunctionDerivativeBasisEntryCalculator,
    L2ProductCorrectionBasisEntryCalculator,
):
    """Calculate int_K (Du_h-g(u_h))*Dv_i, where v_i denotes the FEM basis."""

    def __init__(self, element_space: FiniteElementSpace, quadrature_degree: int):
        local_quadrature = LocalElementQuadrature(quadrature_degree)
        L2ProductFunctionDerivativeBasisEntryCalculator.__init__(
            self, element_space, local_quadrature
        )

        self._build_fast_gradient_approximation()
        self._build_fast_discrete_solution()


class L2ProductCorrectionBasisBuilder(DOFVectorBuilder):
    _entry_calculator: L2ProductCorrectionBasisEntryCalculator

    def build_vector(
        self,
        discrete_solution_dof_vector: np.ndarray,
        gradient_approximation_dof_vector: np.ndarray,
    ) -> np.ndarray:
        self._entry_calculator.set_discrete_solution_dof_vector(
            discrete_solution_dof_vector
        )
        self._entry_calculator.set_gradient_approximation_dof_vector(
            gradient_approximation_dof_vector
        )

        self._entry_calculator.build_left_functions_node_values()

        return DOFVectorBuilder.build_vector(self)
