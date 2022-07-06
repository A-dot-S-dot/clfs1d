from typing import Dict, Sequence

import numpy as np
from math_types import FunctionRealToReal
from numpy import nan, poly1d
from numpy.polynomial import Polynomial
from scipy.interpolate import lagrange

from fem.abstracts import FiniteElement, FiniteElementSpace
from fem.global_element import GlobalFiniteElement
from mesh.interval import Interval
from mesh.mesh import UniformMesh


def discrete_derivative(f: FunctionRealToReal, x: float, eps: float = 1e-7):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


class PiecewiseLagrangeInterpolation:
    _polynomial: Dict[Interval, poly1d]

    def __init__(self):
        self._polynomial = {}

    def add_piecewise_polynomial(
        self,
        interpolation_points: Sequence[float],
        interpolation_values: Sequence[float],
        support: Interval,
    ):
        self._polynomial[support] = lagrange(interpolation_points, interpolation_values)

    def __call__(self, x: float) -> float:
        value = 0
        for support, interpolation in self._polynomial.items():
            if x in support:
                value = interpolation(x)
                break

        return value

    def derivative(self, x: float) -> float:
        value = 0

        for support, interpolation in self._polynomial.items():
            if x in support:
                if support.is_in_boundary(x):
                    value = nan
                else:
                    value = interpolation.deriv()(x)
                break

        return value


class TestFiniteElementSpace(FiniteElementSpace):
    """Represents element space of constant elements. The triangulation is (-1,1)
    and the only basis element is 1."""

    _dimension = 1
    _polynomial_degree = 0
    basis = lambda _, x: 1
    anchor = 0
    _domain = Interval(-1, 1)
    _mesh = UniformMesh(_domain, 1)
    _indices_per_simplex = 1

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def polynomial_degree(self) -> int:
        return self._polynomial_degree

    @property
    def domain(self) -> Interval:
        return self._domain

    @property
    def mesh(self) -> UniformMesh:
        return self._mesh

    @property
    def indices_per_simplex(self) -> int:
        return self._indices_per_simplex

    @property
    def local_basis(self):
        ...

    def get_global_index(self, simplex_index: int, local_index: int) -> int:
        if local_index != 0:
            raise ValueError(f"{local_index} is not zero")
        if simplex_index != 0:
            raise ValueError(f"{simplex_index} is not zero")
        else:
            return 0

    def get_value(self, point: float, dof_vector: np.ndarray) -> float:
        if not self.is_dof_vector(dof_vector):
            raise ValueError("dof_vector must have length 1")

        return dof_vector[0] * self.basis(point)

    def get_value_on_simplex(
        self, point: float, dof_vector: np.ndarray, simplex_index: int
    ) -> float:
        return self.get_value(point, dof_vector)

    def get_derivative(self, _: float, dof_vector: np.ndarray) -> float:
        if not self.is_dof_vector(dof_vector):
            raise ValueError("dof_vector must have length 1")

        return 0

    def get_derivative_on_simplex(
        self, point: float, dof_vector: np.ndarray, simplex_index: int
    ) -> float:
        return self.get_derivative(point, dof_vector)

    def interpolate(self, f: FunctionRealToReal) -> FiniteElement:
        dof_vector = np.array([f(0)])
        return GlobalFiniteElement(self, dof_vector)


# Quadratic lagrange basis elements on {[0,0.5], [0.5,1]}
phi1_00 = Polynomial((1, -2))
phi1_01 = Polynomial((-1, 2))

phi1_10 = Polynomial((0, 2))
phi1_11 = Polynomial((2, -2))

basis1 = [(phi1_00, phi1_01), (phi1_10, phi1_11)]
basis1_derivative = [
    (phi1_00.deriv(), phi1_01.deriv()),
    (phi1_10.deriv(), phi1_11.deriv()),
]

# Quadratic lagrange basis elements on {[0,0.5], [0.5,1]}
phi2_00 = Polynomial((1, -6, 8))
phi2_01 = Polynomial((3, -10, 8))

phi2_10 = Polynomial((0, 8, -16))
phi2_11 = Polynomial((0))

phi2_20 = Polynomial((0, -2, 8))
phi2_21 = Polynomial((6, -14, 8))

phi2_30 = Polynomial((0))
phi2_31 = Polynomial((-8, 24, -16))

basis2 = [
    (phi2_00, phi2_01),
    (phi2_10, phi2_11),
    (phi2_20, phi2_21),
    (phi2_30, phi2_31),
]

basis2_derivative = [
    (phi2_00.deriv(), phi2_01.deriv()),
    (phi2_10.deriv(), phi2_11.deriv()),
    (phi2_20.deriv(), phi2_21.deriv()),
    (phi2_30.deriv(), phi2_31.deriv()),
]
