"""Provides a classes which produces a vector containing L2 products with basis elements or its derivatives."""

from typing import List

import numpy as np
from fem import FiniteElementSpace
from fem.lagrange import LagrangeFiniteElementSpace, LocalLagrangeBasis
from math_types import FunctionRealToReal
from quadrature.local import LocalElementQuadrature

from .assembler import LocalToGlobalVectorAssembler
from .builder import DOFVectorBuilder
from .entry_calculator import DOFVectorEntryCalculator


class L2ProductEntryCalculator(DOFVectorEntryCalculator):
    _left_function_nodes_values_per_simplex: List[List[float]]
    _right_local_functions_nodes_values: List[List[float]]
    _element_space: FiniteElementSpace
    _local_quadrature: LocalElementQuadrature
    _step_length: float

    def __init__(
        self,
        element_space: FiniteElementSpace,
        local_quadrature: LocalElementQuadrature,
    ):
        self._element_space = element_space
        self._local_quadrature = local_quadrature
        self._step_length = element_space.mesh.step_length

    def set_left_function(self, function: FunctionRealToReal):
        self._left_function_nodes_values_per_simplex = []

        for simplex in self._element_space.mesh:
            self._left_function_nodes_values_per_simplex.append(
                [
                    function(simplex.world_coordinates(quadrature_node))
                    for quadrature_node in self._local_quadrature.nodes
                ]
            )

    def __call__(self, simplex_index: int, local_index: int) -> float:
        left_function_nodes_values = self._left_function_nodes_values_per_simplex[
            simplex_index
        ]
        right_function_nodes_values = self._right_local_functions_nodes_values[
            local_index
        ]

        nodes_values = np.array(
            [
                left_function_nodes_values[node_index]
                * right_function_nodes_values[node_index]
                for node_index in range(len(self._local_quadrature.nodes))
            ]
        )

        return self._step_length * np.dot(self._local_quadrature.weights, nodes_values)


class L2ProductFunctionBasisEntryCalculator(L2ProductEntryCalculator):
    _local_basis: LocalLagrangeBasis

    def __init__(
        self,
        element_space: FiniteElementSpace,
        local_quadrature: LocalElementQuadrature,
    ):
        L2ProductEntryCalculator.__init__(self, element_space, local_quadrature)

        if not isinstance(element_space, LagrangeFiniteElementSpace):
            raise NotImplementedError

        self._local_basis = element_space.local_basis
        self._build_right_functions_node_values()

    def _build_right_functions_node_values(self):
        self._right_local_functions_nodes_values = []
        for basis_element in self._local_basis:
            self._right_local_functions_nodes_values.append(
                [basis_element(node) for node in self._local_quadrature.nodes]
            )


class L2ProductFunctionDerivativeBasisEntryCalculator(L2ProductEntryCalculator):
    _local_basis: LocalLagrangeBasis
    _Lambda: np.ndarray

    def __init__(
        self,
        element_space: FiniteElementSpace,
        local_quadrature: LocalElementQuadrature,
    ):
        L2ProductEntryCalculator.__init__(self, element_space, local_quadrature)

        if not isinstance(element_space, LagrangeFiniteElementSpace):
            raise NotImplementedError

        self._local_basis = element_space.local_basis

        self._Lambda = np.array([-1 / self._step_length, 1 / self._step_length])

        self._build_right_functions_node_values()

    def _build_right_functions_node_values(self):
        self._right_local_functions_nodes_values = []
        for basis_element in self._local_basis:
            self._right_local_functions_nodes_values.append(
                [
                    np.dot(self._Lambda, basis_element.derivative(node))
                    for node in self._local_quadrature.nodes
                ]
            )


class L2ProductBuilder(DOFVectorBuilder):
    _element_space: FiniteElementSpace
    _assembler: LocalToGlobalVectorAssembler
    _entry_calculator: L2ProductEntryCalculator

    def setup_entry_calculator(self, left_function: FunctionRealToReal):
        self._entry_calculator.set_left_function(left_function)


class L2ProductBasisBuilder(L2ProductBuilder):
    def __init__(
        self, element_space: FiniteElementSpace, quadrature: LocalElementQuadrature
    ):
        entry_calculator = L2ProductFunctionBasisEntryCalculator(
            element_space, quadrature
        )
        L2ProductBuilder.__init__(self, element_space, entry_calculator)


class L2ProductDerivativeBasisBuilder(L2ProductBuilder):
    def __init__(
        self, element_space: FiniteElementSpace, quadrature: LocalElementQuadrature
    ):
        entry_calculator = L2ProductFunctionDerivativeBasisEntryCalculator(
            element_space, quadrature
        )
        L2ProductBuilder.__init__(self, element_space, entry_calculator)
