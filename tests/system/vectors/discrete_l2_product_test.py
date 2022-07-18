from typing import List
from unittest import TestCase

import numpy as np
from fem import FastFunction
from fem.fast_element import FastFiniteElement
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh.mesh import Interval, UniformMesh
from quadrature.local import LocalElementQuadrature
from scipy.integrate import quad
from system.vectors.discrete_l2_product import (
    DiscreteL2ProductBuilder,
    DiscreteGradientL2ProductBuilder,
)
from ...test_helper.lagrange_basis_elements import *


class TestL2ProductLinearBasisBuilder(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    quadrature = LocalElementQuadrature(2)
    test_functions = [lambda _: 1, lambda x: x, lambda x: x**2 - 2 * x]
    test_fast_functions = [
        FastFunction(lambda _: 1, lambda _: 0, mesh, quadrature.nodes),
        FastFunction(lambda x: x, lambda _: 1, mesh, quadrature.nodes),
        FastFunction(
            lambda x: x**2 - 2 * x, lambda x: 2 * x - 2, mesh, quadrature.nodes
        ),
    ]
    test_functions_strings = ["1", "x", "x²-2x"]
    basis = basis1
    discrete_l2_product_builder = DiscreteL2ProductBuilder(element_space, quadrature)
    expected_dof_vectors: List[np.ndarray]

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)

        self._build_expected_values()

    def _build_expected_values(self):
        self.expected_dof_vectors = []
        for function in self.test_functions:
            expected_dof_vector = np.zeros(self.element_space.dimension)

            for i in range(self.element_space.dimension):
                integrand0 = lambda x: self.basis[i][0](x) * function(x)
                integrand1 = lambda x: self.basis[i][1](x) * function(x)

                expected_dof_vector[i] = (
                    quad(integrand0, 0, 0.5)[0] + quad(integrand1, 0.5, 1)[0]
                )

            self.expected_dof_vectors.append(expected_dof_vector)

    def test_build_vector(self):
        for fast_function, functions_strings, expected_l2_product in zip(
            self.test_fast_functions,
            self.test_functions_strings,
            self.expected_dof_vectors,
        ):
            self.discrete_l2_product_builder.set_left_function(fast_function)
            l2_product = self.discrete_l2_product_builder.build_dof()
            for i in range(self.element_space.dimension):
                self.assertAlmostEqual(
                    l2_product[i],
                    expected_l2_product[i],
                    msg=f"entry={i}, f(x)={functions_strings}",
                )


class TestL2ProductLinearDerivativeBasisBuilder(TestL2ProductLinearBasisBuilder):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    quadrature = LocalElementQuadrature(2)
    basis = basis1_derivative
    discrete_l2_product_builder = DiscreteGradientL2ProductBuilder(
        element_space, quadrature
    )


class TestL2ProductQuadraticBasisBuilder(TestL2ProductLinearBasisBuilder):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    quadrature = LocalElementQuadrature(3)
    test_fast_functions = [
        FastFunction(lambda _: 1, lambda _: 0, mesh, quadrature.nodes),
        FastFunction(lambda x: x, lambda _: 1, mesh, quadrature.nodes),
        FastFunction(
            lambda x: x**2 - 2 * x, lambda x: 2 * x - 2, mesh, quadrature.nodes
        ),
    ]
    basis = basis2
    discrete_l2_product_builder = DiscreteL2ProductBuilder(element_space, quadrature)


class TestL2ProductQuadraticDerivativeBasisBuilder(TestL2ProductLinearBasisBuilder):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    quadrature = LocalElementQuadrature(3)
    test_fast_functions = [
        FastFunction(lambda _: 1, lambda _: 0, mesh, quadrature.nodes),
        FastFunction(lambda x: x, lambda _: 1, mesh, quadrature.nodes),
        FastFunction(
            lambda x: x**2 - 2 * x, lambda x: 2 * x - 2, mesh, quadrature.nodes
        ),
    ]
    basis = basis2_derivative
    discrete_l2_product_builder = DiscreteGradientL2ProductBuilder(
        element_space, quadrature
    )
