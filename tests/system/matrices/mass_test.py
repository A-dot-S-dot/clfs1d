from unittest import TestCase

import numpy as np
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh.mesh import Interval, UniformMesh
from system.matrices.mass import MassEntryCalculator, MassMatrix


class TestMassEntryCalculator(TestCase):
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, 1)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    expected_local_matrix = interval.length / 6 * np.array([[2, 1], [1, 2]])
    mass_entry_calculator = MassEntryCalculator(element_space)

    def test_entries(self):
        for i in range(self.element_space.indices_per_simplex):
            for j in range(self.element_space.indices_per_simplex):
                self.assertAlmostEqual(
                    self.mass_entry_calculator(0, i, j),
                    self.expected_local_matrix[i, j],
                )


class TestMass(TestCase):
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, 4)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    mass = MassMatrix(element_space)
    expected_mass = (
        mesh.step_length
        / 6
        * np.array([[4, 1, 0, 1], [1, 4, 1, 0], [0, 1, 4, 1], [1, 0, 1, 4]])
    )

    def test_entries(self):
        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertAlmostEqual(self.mass[i, j], self.expected_mass[i, j])
