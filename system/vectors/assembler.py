import numpy as np
from fem import FiniteElementSpace

from .entry_calculator import DOFVectorEntryCalculator


class LocalToGlobalVectorAssembler:
    """Fills a dof vector using from local to global principles."""

    _element_space: FiniteElementSpace
    _entry_calculator: DOFVectorEntryCalculator

    def __init__(
        self,
        element_space: FiniteElementSpace,
        entry_calculator: DOFVectorEntryCalculator,
    ):
        self._element_space = element_space
        self._entry_calculator = entry_calculator

    def fill_entries(self, vector: np.ndarray):
        for simplex_index in range(len(self._element_space.mesh)):
            for local_index in range(self._element_space.indices_per_simplex):
                global_index = self._element_space.get_global_index(
                    simplex_index, local_index
                )

                vector[global_index] += self._entry_calculator(
                    simplex_index, local_index
                )
