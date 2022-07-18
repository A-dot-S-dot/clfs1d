import numpy as np
from fem import FiniteElementSpace

from .entry_calculator import DOFEntryCalculator


class LocalToGlobalVectorAssembler:
    """Fills a DOF vector using from local to global principles.

    We loop through each element of the mesh and its local indices, calculate
    something for that and add it to the related entry of the global vector.

    """

    _element_space: FiniteElementSpace
    _entry_calculator: DOFEntryCalculator

    def __init__(
        self,
        element_space: FiniteElementSpace,
        entry_calculator: DOFEntryCalculator,
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
