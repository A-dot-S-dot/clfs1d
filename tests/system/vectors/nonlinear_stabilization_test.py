from unittest import TestCase

import numpy as np
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh.interval import Interval
from mesh.mesh import UniformMesh
from system.vectors.nonlinear_stabilization import NonlinearStabilizationBuilder
from system.vectors.averaged_gradient import AveragedGradientDOFBuilder


class TestLinearNonlinearStabilization(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    stabilization_parameter = 0.5
    gradient_builder = AveragedGradientDOFBuilder(element_space)
    builder = NonlinearStabilizationBuilder(
        element_space, stabilization_parameter, gradient_builder
    )
    dof_vectors = [np.array([1, 0]), np.array([0, 1])]
    expected_nonlinear_stabilization_dof_vectors = [
        np.array([stabilization_parameter, -stabilization_parameter]),
        np.array([-stabilization_parameter, stabilization_parameter]),
    ]

    def test_nonlinear_stabilization(self):
        for dof_vector, expected_nonlinear_stabilization in zip(
            self.dof_vectors, self.expected_nonlinear_stabilization_dof_vectors
        ):

            nonlinear_stabilization = self.builder.build_dof(dof_vector)
            for stabilization_dof, expected_stabilization_dof in zip(
                nonlinear_stabilization, expected_nonlinear_stabilization
            ):
                self.assertAlmostEqual(stabilization_dof, expected_stabilization_dof)


class TestQuadraticNonlinearStabilization(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    stabilization_parameter = 0.5
    gradient_builder = AveragedGradientDOFBuilder(element_space)
    builder = NonlinearStabilizationBuilder(
        element_space, stabilization_parameter, gradient_builder
    )
    uh_dof_vector = np.array([1, 0, 0, 0])
    expected_nonlinear_stabilization_dof_vector = np.array(
        [
            0.5425107989420075,
            -0.32826937403102313,
            0.11402794912003904,
            -0.32826937403102313,
        ]
    )

    def test_nonlinear_stabilization(self):
        nonlinear_stabilization = self.builder.build_dof(self.uh_dof_vector)

        for i in range(len(nonlinear_stabilization)):
            self.assertAlmostEqual(
                nonlinear_stabilization[i],
                self.expected_nonlinear_stabilization_dof_vector[i],
                delta=0.1,
            )
