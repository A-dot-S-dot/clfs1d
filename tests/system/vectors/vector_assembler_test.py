from unittest import TestCase

import numpy as np
from system.vectors.assembler import LocalToGlobalVectorAssembler
from system.vectors.entry_calculator import DOFEntryCalculator

from ...test_helper import TestFiniteElementSpace


class TestVectorEntryCalculator(DOFEntryCalculator):
    def __call__(self, simplex_index: int, local_index: int) -> float:
        return 2


class TestVectorAssembler(TestCase):
    element_space = TestFiniteElementSpace()
    entry_calculator = TestVectorEntryCalculator()

    def test_fill_entries(self):
        assembler = LocalToGlobalVectorAssembler(
            self.element_space, self.entry_calculator
        )
        system_vector = np.array([0])

        assembler.fill_entries(system_vector)

        self.assertEqual(system_vector[0], 2)
