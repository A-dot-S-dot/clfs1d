from unittest import TestCase

import numpy as np
from quadrature.local import LocalElementQuadrature
from mesh.mesh import Interval, UniformMesh
from mesh.transformation import AffineTransformation

from fem.fast_element import (
    AnchorNodesFastFiniteElement,
    FastFunction,
    FastLocalElement,
    QuadratureFastFiniteElement,
)
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from fem.lagrange.local_lagrange import LocalLagrangeBasis

from ..test_helper.lagrange_basis_elements import *


class TestFastFunction(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    local_quadrature = LocalElementQuadrature(2)
    nodes = local_quadrature.nodes
    test_functions = [lambda x: x, lambda x: np.sin(x)]
    test_derivatives = [lambda _: 1, lambda x: np.cos(x)]
    test_function_strings = ["x", "sin(x)"]
    affine_transformation = AffineTransformation()

    def test_fast_function_values(self):
        for function, derivative, f_string in zip(
            self.test_functions, self.test_derivatives, self.test_function_strings
        ):
            fast_function = FastFunction(function, derivative, self.mesh, self.nodes)

            for simplex_index, simplex in enumerate(self.mesh):
                for node_index, node in enumerate(self.nodes):
                    point = self.affine_transformation(node, simplex)
                    self.assertAlmostEqual(
                        fast_function.value(simplex_index, node_index),
                        function(point),
                        msg=f"K={simplex}, x={point}, f(x)={f_string}",
                    )

    def test_fast_function_derivatives(self):
        for function, derivative, f_string in zip(
            self.test_functions, self.test_derivatives, self.test_function_strings
        ):
            fast_function = FastFunction(function, derivative, self.mesh, self.nodes)

            for simplex_index, simplex in enumerate(self.mesh):
                for node_index, node in enumerate(self.nodes):
                    point = self.affine_transformation(node, simplex)
                    self.assertAlmostEqual(
                        fast_function.derivative(simplex_index, node_index),
                        derivative(point),
                        msg=f"K={simplex}, x={point}, f(x)={f_string}",
                    )


class TestFastLocalElement(TestCase):
    local_basis = LocalLagrangeBasis(1)
    nodes = [0, 0.5, 1]
    fast_element = FastLocalElement(local_basis[0])
    expected_values = [1, 0.5, 0]
    expected_derivatives = [-1, -1, -1]

    def test_fast_local_element_values(self):
        self.fast_element.set_values(*self.nodes)

        for node_index, node in enumerate(self.nodes):
            self.assertAlmostEqual(
                self.fast_element.value(node_index),
                self.expected_values[node_index],
                msg=f"node={node}",
            )

    def test_fast_local_element_derivatives(self):
        self.fast_element.set_derivatives(*self.nodes)

        for node_index, node in enumerate(self.nodes):
            self.assertAlmostEqual(
                self.fast_element.derivative(node_index),
                self.expected_derivatives[node_index],
                msg=f"node={node}",
            )


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
    affine_transformation = AffineTransformation()

    def test_fast_element_values(self):
        self.fast_element.set_dof(self.test_dof)
        self.fast_element.set_values()

        for simplex_index, simplex in enumerate(self.mesh):
            for node_index, node in enumerate(self.nodes):
                self.assertAlmostEqual(
                    self.fast_element.value(simplex_index, node_index),
                    self.test_finite_element[simplex_index](
                        self.affine_transformation(node, simplex)
                    ),
                )

    def test_fast_element_derivative(self):
        self.fast_element.set_dof(self.test_dof)
        self.fast_element.set_derivatives()

        for simplex, simplex_index in zip(self.mesh, range(len(self.mesh))):
            for node, node_index in zip(self.nodes, range(len(self.nodes))):
                self.assertAlmostEqual(
                    self.fast_element.derivative(simplex_index, node_index),
                    self.test_finite_element_derivative[simplex_index](
                        self.affine_transformation(node, simplex)
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
