from unittest import TestCase

import numpy as np
from system.matrices.assembler import (
    LocalToGlobalMatrixAssembler,
    SystemMatrixEntryCalculator,
)

from ...test_helper import TestFiniteElementSpace


class TestMatrixEntryCalculator(SystemMatrixEntryCalculator):
    def __call__(
        self, simplex_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        return 2


class TestLocalToGlobalMatrixAssembler(TestCase):
    element_space = TestFiniteElementSpace()
    entry_calculator = TestMatrixEntryCalculator()

    def test_fill_entries(self):
        assembler = LocalToGlobalMatrixAssembler(
            self.element_space, self.entry_calculator
        )
        system_matrix = np.array([[0]])

        assembler.fill_entries(system_matrix)

        self.assertEqual(system_matrix[0, 0], 2)
