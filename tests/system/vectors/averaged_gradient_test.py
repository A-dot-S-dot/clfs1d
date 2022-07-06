from unittest import TestCase

import numpy as np
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh.interval import Interval
from mesh.mesh import UniformMesh
from system.vectors.averaged_gradient import AveragedGradientBuilder


class TestLinearAveragedGradient(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 3)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    builder = AveragedGradientBuilder(element_space)
    dof_vectors = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    expected_averaged_gradient_dof_vectors = [
        np.array([0, -3 / 2, 3 / 2]),
        np.array([3 / 2, 0, -3 / 2]),
    ]

    def test_averaged_gradient(self):
        for dof_vector, expected_averaged_gradient in zip(
            self.dof_vectors, self.expected_averaged_gradient_dof_vectors
        ):
            averaged_gradient = self.builder.build_vector(dof_vector)
            self.assertTupleEqual(
                tuple(averaged_gradient), tuple(expected_averaged_gradient)
            )


class TestQuadraticAveragedGradient(TestLinearAveragedGradient):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    builder = AveragedGradientBuilder(element_space)
    dof_vectors = [
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
    ]
    expected_averaged_gradient_dof_vectors = [
        np.array([0, -2, 0, 2]),
        np.array([4, 0, -4, 0]),
        np.array([0, 2, 0, -2]),
        np.array([-4, 0, 4, 0]),
    ]
