from typing import List
from unittest import TestCase

import numpy as np
from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh.mesh import Interval, UniformMesh
from quadrature.local import LocalElementQuadrature
from scipy.integrate import quad
from system.vectors.l2_product_function_basis import (
    L2ProductBasisBuilder,
    L2ProductDerivativeBasisBuilder,
)
from ...test_helper import *


class TestL2ProductLinearBasisBuilder(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    quadrature = LocalElementQuadrature(2)
    test_functions = [lambda x: 1, lambda x: x, lambda x: x**2 - 2 * x]
    test_functions_strings = ["1", "x", "x²-2x"]
    basis = basis1
    l2_product_builder = L2ProductBasisBuilder(element_space, quadrature)
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
        for function, functions_strings, expected_l2_product in zip(
            self.test_functions, self.test_functions_strings, self.expected_dof_vectors
        ):
            self.l2_product_builder.setup_entry_calculator(function)
            l2_product = self.l2_product_builder.build_vector()
            for i in range(self.element_space.dimension):
                self.assertAlmostEqual(
                    l2_product[i],
                    expected_l2_product[i],
                    msg=f"index={i}, f(x)={functions_strings}",
                )


class TestL2ProductLinearDerivativeBasisBuilder(TestL2ProductLinearBasisBuilder):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    quadrature = LocalElementQuadrature(2)
    basis = basis1_derivative
    l2_product_builder = L2ProductDerivativeBasisBuilder(element_space, quadrature)


class TestL2ProductQuadraticBasisBuilder(TestL2ProductLinearBasisBuilder):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    quadrature = LocalElementQuadrature(3)
    basis = basis2
    l2_product_builder = L2ProductBasisBuilder(element_space, quadrature)


class TestL2ProductQuadraticDerivativeBasisBuilder(TestL2ProductLinearBasisBuilder):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    quadrature = LocalElementQuadrature(3)
    basis = basis2_derivative
    l2_product_builder = L2ProductDerivativeBasisBuilder(element_space, quadrature)
