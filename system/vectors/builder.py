import numpy as np
from typing import Any
from fem import FiniteElementSpace

from .assembler import LocalToGlobalVectorAssembler
from .entry_calculator import DOFVectorEntryCalculator


class DOFVectorBuilder:
    """Returns for a given finite element (represented as a dof vector) a dof vector
    which entries are built with an `VectorEntryCalculator` instance.

    """

    _assembler: LocalToGlobalVectorAssembler
    _element_space: FiniteElementSpace
    _entry_calculator: DOFVectorEntryCalculator

    def __init__(
        self,
        element_space: FiniteElementSpace,
        entry_calculator: DOFVectorEntryCalculator,
    ):
        self._assembler = LocalToGlobalVectorAssembler(element_space, entry_calculator)
        self._element_space = element_space
        self._entry_calculator = entry_calculator

    def build_vector(self, *args: Any) -> np.ndarray:
        dof_vector = np.zeros(self._element_space.dimension)
        self._assembler.fill_entries(dof_vector)

        return dof_vector
