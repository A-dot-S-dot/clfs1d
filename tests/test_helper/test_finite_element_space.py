import numpy as np
from fem import FiniteElement, FiniteElementSpace, GlobalFiniteElement
from math_types import FunctionRealToReal
from mesh import Interval, UniformMesh


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
