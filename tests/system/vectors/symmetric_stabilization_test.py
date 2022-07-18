from unittest import TestCase

import numpy as np
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh.interval import Interval
from mesh.mesh import UniformMesh
from scipy.integrate import quad
from system.vectors.averaged_gradient import AveragedGradientDOFBuilder
from system.vectors.symmetric_stabilization import SymmetricStabilizationBuilder

from ...test_helper.lagrange_basis_elements import *


class TestLinearSymmetricStabilization(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    stabilization_parameter = 0.5
    gradient_builder = AveragedGradientDOFBuilder(element_space)
    builder = SymmetricStabilizationBuilder(
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


class TestQuadraticSymmetricStabilization(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    stabilization_parameter = 0.5
    gradient_builder = AveragedGradientDOFBuilder(element_space)
    builder = SymmetricStabilizationBuilder(
        element_space, stabilization_parameter, gradient_builder
    )
    uh_dof_vector = np.array([1, 0, 0, 0])
    expected_nonlinear_stabilization_dof_vector: np.ndarray

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        self._build_expected_stabilization_dof_vector()

    def _build_expected_stabilization_dof_vector(self):
        g00 = -2 * phi2_10 + 2 * phi2_30
        g01 = -2 * phi2_11 + 2 * phi2_31

        g10 = 4 * phi2_00 - 4 * phi2_20
        g11 = 4 * phi2_01 - 4 * phi2_21

        g20 = 2 * phi2_10 - 2 * phi2_30
        g21 = 2 * phi2_11 - 2 * phi2_31

        g30 = -g10
        g31 = -g11

        g = [(g00, g01), (g10, g11), (g20, g21), (g30, g31)]

        s = [
            (phi2_i0.deriv() - gi0, phi2_i1.deriv() - gi1)
            for (phi2_i0, phi2_i1), (gi0, gi1) in zip(basis2, g)
        ]

        self.expected_nonlinear_stabilization_dof_vector = (
            self.stabilization_parameter
            / 8
            * np.array(
                [
                    quad(s[0][0] * si0, 0, 0.5)[0] + quad(s[0][1] * si1, 0.5, 1)[0]
                    for si0, si1 in s
                ]
            )
        )

    def test_nonlinear_stabilization(self):
        nonlinear_stabilization = self.builder.build_dof(self.uh_dof_vector)

        for i in range(len(nonlinear_stabilization)):
            self.assertAlmostEqual(
                nonlinear_stabilization[i],
                self.expected_nonlinear_stabilization_dof_vector[i],
            )
