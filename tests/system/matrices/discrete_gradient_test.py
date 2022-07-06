from unittest import TestCase

import numpy as np
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh.mesh import Interval, UniformMesh

from system.matrices.discrete_gradient import (
    DiscreteGradientEntryCalculator,
    DiscreteGradientMatrix,
)


class TestDiscreteGradientEntryCalculator(TestCase):
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    expected_local_matrix = 1 / 2 * np.array([[-1, 1], [-1, 1]])
    discrete_gradient_entry_calculator = DiscreteGradientEntryCalculator(element_space)

    def test_entries(self):
        for i in range(self.element_space.indices_per_simplex):
            for j in range(self.element_space.indices_per_simplex):
                self.assertAlmostEqual(
                    self.discrete_gradient_entry_calculator(0, i, j),
                    self.expected_local_matrix[i, j],
                )


class TestLinearDiscreteGradient(TestCase):
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, 4)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    discrete_gradient = DiscreteGradientMatrix(element_space)
    expected_discrete_gradient = (
        1 / 2 * np.array([[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]])
    )

    def test_entries(self):
        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertAlmostEqual(
                    self.discrete_gradient[i, j], self.expected_discrete_gradient[i, j]
                )


class TestQuadraticDiscreteGradient(TestLinearDiscreteGradient):
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    discrete_gradient = DiscreteGradientMatrix(element_space)
    expected_discrete_gradient = np.array(
        [
            [0, 2 / 3, 0, -2 / 3],
            [-2 / 3, 0, 2 / 3, 0],
            [0, -2 / 3, 0, 2 / 3],
            [2 / 3, 0, -2 / 3, 0],
        ]
    )
