from abc import ABC, abstractmethod

from fem import FiniteElementSpace

from .entry_calculator import SystemMatrixEntryCalculator


class MatrixAssembler(ABC):
    """Assembles a given matrix."""

    @abstractmethod
    def fill_entries(self, system_matrix):
        ...


class LocalToGlobalMatrixAssembler(MatrixAssembler):
    _element_space: FiniteElementSpace
    _matrix_entry_calculator: SystemMatrixEntryCalculator

    def __init__(
        self,
        element_space: FiniteElementSpace,
        matrix_entry_calculator: SystemMatrixEntryCalculator,
    ):
        self._element_space = element_space
        self._matrix_entry_calculator = matrix_entry_calculator

    def fill_entries(self, system_matrix):
        for simplex_index in range(len(self._element_space.mesh)):
            for local_index_1 in range(self._element_space.indices_per_simplex):
                for local_index_2 in range(self._element_space.indices_per_simplex):
                    global_index_1 = self._element_space.get_global_index(
                        simplex_index, local_index_1
                    )
                    global_index_2 = self._element_space.get_global_index(
                        simplex_index, local_index_2
                    )

                    system_matrix[
                        global_index_1, global_index_2
                    ] += self._matrix_entry_calculator(
                        simplex_index, local_index_1, local_index_2
                    )
