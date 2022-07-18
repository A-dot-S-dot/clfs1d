from unittest import TestCase

import numpy as np
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh.interval import Interval
from mesh.mesh import UniformMesh
from system.vectors.averaged_gradient import AveragedGradientDOFBuilder


class TestLinearAveragedGradient(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 3)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    builder = AveragedGradientDOFBuilder(element_space)
    test_element_dofs = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    expected_averaged_gradient_dofs = [
        np.array([0, -3 / 2, 3 / 2]),
        np.array([3 / 2, 0, -3 / 2]),
    ]

    def test_averaged_gradient(self):
        for test_element_dof, expected_averaged_gradient_dof in zip(
            self.test_element_dofs, self.expected_averaged_gradient_dofs
        ):
            averaged_gradient_dof = self.builder.build_dof(test_element_dof)
            for test_entry, expected_entry in zip(
                averaged_gradient_dof, expected_averaged_gradient_dof
            ):
                self.assertAlmostEqual(test_entry, expected_entry)


class TestQuadraticAveragedGradient(TestLinearAveragedGradient):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    builder = AveragedGradientDOFBuilder(element_space)
    test_element_dofs = [
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
    ]
    expected_averaged_gradient_dofs = [
        np.array([0, -2, 0, 2]),
        np.array([4, 0, -4, 0]),
        np.array([0, 2, 0, -2]),
        np.array([-4, 0, 4, 0]),
    ]
