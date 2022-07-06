from unittest import TestCase

import numpy as np
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh.mesh import Interval, UniformMesh
from system.matrices.gradient_approximation import GradientApproximationMatrix
from system.matrices.mass import MassMatrix
from system.vectors.averaged_gradient import AveragedGradientBuilder
from system.vectors.l2_projection_gradient import L2ProjectionGradientBuilder


class TestLinearAveragedGradientApproximation(TestCase):
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    gradient_approximation_builder = AveragedGradientBuilder(element_space)
    gradient_approximation = GradientApproximationMatrix(
        element_space, gradient_approximation_builder
    )
    expected_gradient_approximation = np.array([[0, 0], [0, 0]])

    def test_entries(self):
        for i in range(self.element_space.dimension):
            for j in range(self.element_space.dimension):
                self.assertAlmostEqual(
                    self.gradient_approximation[i, j],
                    self.expected_gradient_approximation[i, j],
                    msg=f"indices=({i},{j})",
                )


class TestLinearL2ProjectionGradientApproximation(
    TestLinearAveragedGradientApproximation
):
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    mass = MassMatrix(element_space, build_inverse=True)
    gradient_approximation_builder = L2ProjectionGradientBuilder(element_space, mass)
    gradient_approximation = GradientApproximationMatrix(
        element_space, gradient_approximation_builder
    )
    expected_gradient_approximation = np.array([[0, 0], [0, 0]])


class TestQuadraticAveragedGradientApproximation(
    TestLinearAveragedGradientApproximation
):
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    gradient_approximation_builder = AveragedGradientBuilder(element_space)
    gradient_approximation = GradientApproximationMatrix(
        element_space, gradient_approximation_builder
    )
    expected_gradient_approximation = np.array(
        [[0, -2, 0, 2], [4, 0, -4, 0], [0, 2, 0, -2], [-4, 0, 4, 0]]
    )


class TestQuadraticL2ProjectionGradientApproximation(
    TestLinearAveragedGradientApproximation
):
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    mass = MassMatrix(element_space, build_inverse=True)
    gradient_approximation_builder = L2ProjectionGradientBuilder(element_space, mass)
    gradient_approximation = GradientApproximationMatrix(
        element_space, gradient_approximation_builder
    )
    expected_gradient_approximation = np.array(
        [[0, -2.5, 0, 2.5], [4, 0, -4, 0], [0, 2.5, 0, -2.5], [-4, 0, 4, 0]]
    )
