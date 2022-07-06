from unittest import TestCase

from system.vectors.l2_projection_gradient import L2ProjectionGradientBuilder
from system.matrices.mass import MassMatrix

from mesh.mesh import Interval, UniformMesh
from fem.lagrange.lagrange import LagrangeFiniteElementSpace

import numpy as np


class TestLinearL2ProjectionGradient(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    mass = MassMatrix(element_space, build_inverse=True)
    dof_builder = L2ProjectionGradientBuilder(element_space, mass)
    test_dof_vectors = [np.array([1, 0]), np.array([0, 1])]
    expected_gradients = [[0, 0], [0, 0]]

    def test_build_vector(self):
        for dof_vector, expected_gradient in zip(
            self.test_dof_vectors, self.expected_gradients
        ):
            gradient = self.dof_builder.build_vector(dof_vector)
            for i in range(len(gradient)):
                self.assertAlmostEqual(
                    gradient[i],
                    expected_gradient[i],
                    msg=f"dof_vector={dof_vector}, index={i}",
                )


class TestQuadraticL2ProjectionGradient(TestLinearL2ProjectionGradient):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    mass = MassMatrix(element_space, build_inverse=True)
    dof_builder = L2ProjectionGradientBuilder(element_space, mass)
    test_dof_vectors = [
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
    ]
    expected_gradients = [
        np.array([0, -2.5, 0, 2.5]),
        np.array([4, 0, -4, 0]),
        np.array([0, 2.5, 0, -2.5]),
        np.array([-4, 0, 4, 0]),
    ]
