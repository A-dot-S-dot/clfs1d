from unittest import TestCase

from system.matrices import SystemMatrix, MatrixAssembler
import numpy as np


class TestAssembler(MatrixAssembler):
    def fill_entries(self, matrix: SystemMatrix):
        matrix[0, 0] = matrix[1, 1] = 0
        matrix[0, 1] = matrix[1, 0] = 1


class TestSystemMatrix(TestCase):
    assembler = TestAssembler()
    matrix = SystemMatrix(2, assembler)

    def test_assemble(self):
        for i in range(2):
            for j in range(2):
                if i == j:
                    matrix_entry = 0
                else:
                    matrix_entry = 1
                self.assertEqual(self.matrix[i, j], matrix_entry)

    def test_inverse(self):
        b = np.array([1, 0])
        x = self.matrix.inverse(b)

        self.assertAlmostEqual(x[0], 0)
        self.assertAlmostEqual(x[1], 1)

    def test_permanent_inverse(self):
        matrix = SystemMatrix(2, self.assembler, build_inverse=True)

        b = np.array([1, 0])
        x = matrix.inverse(b)

        self.assertAlmostEqual(x[0], 0)
        self.assertAlmostEqual(x[1], 1)
