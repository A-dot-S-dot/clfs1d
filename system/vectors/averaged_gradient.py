"""This module provides averaged gradients for finite elements.

The main idea is to approximate the gradient of a discrete solution uh with a
finite element function. Its values on a lagrange node is the mean value of the
gradients of uh on every simplex which contains this node. i.e. in one dimension
and uniform meshes we have

    g_j(uh) = 1/2*( D+uh + D-uh ).

See 'Entropy conservation property and entropy stabilization of high-order
continuous Galerkin approximations to scalar conservation laws', D. Kuzmin, M.
Quezada de Luna, 2020, p. 3

"""

import numpy as np
from fem import FiniteElementSpace
from fem.fast_element import AnchorNodesFastFiniteElement

from .builder import DOFVectorBuilder
from .entry_calculator import DOFVectorEntryCalculator


class AveragedGradientEntryCalculator(DOFVectorEntryCalculator):
    _element_space: FiniteElementSpace
    _fast_element: AnchorNodesFastFiniteElement
    _current_simplex_index: int

    def __init__(self, element_space: FiniteElementSpace):
        self._element_space = element_space
        self._fast_element = AnchorNodesFastFiniteElement(
            element_space,
        )
        self._fast_element.add_derivatives()
        self._current_simplex_index = -1

    def set_dof_vector(self, dof_vector: np.ndarray):
        self._fast_element.set_dof_vector(dof_vector)

    def __call__(self, simplex_index: int, local_index: int) -> float:
        weight = self._calculate_weight(local_index)
        entry = self._build_entry(simplex_index, local_index)

        return weight * entry

    def _calculate_weight(self, local_index: int) -> float:
        if local_index in [0, self._element_space.indices_per_simplex - 1]:
            return 1 / 2
        else:
            return 1

    def _build_entry(self, simplex_index: int, local_index: int) -> float:
        return self._fast_element.derivative_on_simplex(simplex_index, local_index)


class AveragedGradientBuilder(DOFVectorBuilder):
    _entry_calculator: AveragedGradientEntryCalculator

    def __init__(self, element_space: FiniteElementSpace):
        entry_calculator = AveragedGradientEntryCalculator(element_space)
        DOFVectorBuilder.__init__(self, element_space, entry_calculator)

    def build_vector(self, dof_vector_to_approximate: np.ndarray) -> np.ndarray:
        self._entry_calculator.set_dof_vector(dof_vector_to_approximate)
        return super().build_vector()
