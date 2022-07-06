from unittest import TestCase

import numpy as np
from quadrature.local import LocalElementQuadrature

from fem.fast_element import AnchorNodesFastFiniteElement, QuadratureFastFiniteElement
from fem.lagrange.lagrange import LagrangeFiniteElementSpace

from ..test_helper import *


class TestLinearQuadratureFastFiniteElement(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    local_quadrature = LocalElementQuadrature(2)
    fast_element = QuadratureFastFiniteElement(element_space, local_quadrature)
    nodes = local_quadrature.nodes
    test_dof = np.array([1, 0])
    test_finite_element = [phi1_00, phi1_01]
    test_finite_element_derivative = [phi1_00.deriv(), phi1_01.deriv()]

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)

        self.fast_element.set_dof_vector(self.test_dof)

    def test_fast_element_values(self):
        self.fast_element.add_values()

        for simplex, simplex_index in zip(self.mesh, range(len(self.mesh))):
            for node, node_index in zip(self.nodes, range(len(self.nodes))):
                self.assertAlmostEqual(
                    self.fast_element.value_on_simplex(simplex_index, node_index),
                    self.test_finite_element[simplex_index](
                        simplex.world_coordinates(node)
                    ),
                )

    def test_fast_element_derivative(self):
        self.fast_element.add_derivatives()

        for simplex, simplex_index in zip(self.mesh, range(len(self.mesh))):
            for node, node_index in zip(self.nodes, range(len(self.nodes))):
                self.assertAlmostEqual(
                    self.fast_element.derivative_on_simplex(simplex_index, node_index),
                    self.test_finite_element_derivative[simplex_index](
                        simplex.world_coordinates(node)
                    ),
                )


class TestQuadraticQuadratureFastFiniteElement(TestLinearQuadratureFastFiniteElement):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    local_quadrature = LocalElementQuadrature(3)
    fast_element = QuadratureFastFiniteElement(element_space, local_quadrature)
    nodes = local_quadrature.nodes
    test_dof = np.array([1, 0, 0, 0])
    test_finite_element = [phi2_00, phi2_01]
    test_finite_element_derivative = [phi2_00.deriv(), phi2_01.deriv()]


class TestLinearAnchorNodesFastFiniteElement(TestLinearQuadratureFastFiniteElement):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    local_quadrature = LocalElementQuadrature(2)
    fast_element = AnchorNodesFastFiniteElement(element_space)
    nodes = fast_element._anchor_nodes
    test_dof = np.array([1, 0])
    test_finite_element = [phi1_00, phi1_01]
    test_finite_element_derivative = [phi1_00.deriv(), phi1_01.deriv()]


class TestQuadraticAnchorNodesFastFiniteElement(TestLinearQuadratureFastFiniteElement):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    local_quadrature = LocalElementQuadrature(2)
    fast_element = AnchorNodesFastFiniteElement(element_space)
    nodes = fast_element._anchor_nodes
    test_dof = np.array([1, 0, 0, 0])
    test_finite_element = [phi2_00, phi2_01]
    test_finite_element_derivative = [phi2_00.deriv(), phi2_01.deriv()]
