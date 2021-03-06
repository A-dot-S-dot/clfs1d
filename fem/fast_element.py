from abc import ABC, abstractmethod
from typing import List
from math_types import FunctionRealToReal

import numpy as np
from mesh.transformation import AffineTransformation
from mesh import UniformMesh
from quadrature.local import LocalElementQuadrature

from .abstracts import FiniteElementSpace, LocalFiniteElement


class FastMapping(ABC):
    """Returns a value or derivative for certain simplex indices of a mesh and a
    point which is also accessed by an index.

    """

    @abstractmethod
    def value(self, simplex_index: int, local_index: int) -> float:
        ...

    @abstractmethod
    def derivative(self, simplex_index: int, local_index: int) -> float:
        ...


class FastFunction(FastMapping):
    _function: FunctionRealToReal
    _derivative: FunctionRealToReal
    _mesh: UniformMesh
    _local_points: List[float]
    _affine_transformation: AffineTransformation

    def __init__(
        self,
        function: FunctionRealToReal,
        derivative: FunctionRealToReal,
        mesh: UniformMesh,
        local_points: List[float],
    ):
        self._function = function
        self._derivative = derivative
        self._mesh = mesh
        self._local_points = local_points
        self._affine_transformation = AffineTransformation()

    def value(self, simplex_index: int, local_index: int) -> float:
        simplex = self._mesh[simplex_index]
        local_point = self._local_points[local_index]

        return self._function(self._affine_transformation(local_point, simplex))

    def derivative(self, simplex_index: int, local_index: int) -> float:
        simplex = self._mesh[simplex_index]
        local_point = self._local_points[local_index]

        return self._derivative(self._affine_transformation(local_point, simplex))


class FastLocalElement:
    """This element is made for better performance while calculating values and
    derivatives of local Lagrange elements.

    Invoke 'set_values' and 'set_derivatives' for adding values and derivatives.

    The desired value can be accessed via 'value' or 'derivative'.

    """

    _local_element: LocalFiniteElement
    _values: List[float]
    _derivatives: List[float]

    def __init__(self, local_element: LocalFiniteElement):
        self._local_element = local_element

    def set_values(self, *local_points: float):
        self._values = [self._local_element(point) for point in local_points]

    def set_derivatives(self, *local_points: float):
        self._derivatives = [
            self._local_element.derivative(point) for point in local_points
        ]

    def value(self, local_index: int) -> float:
        return self._values[local_index]

    def derivative(self, local_index: int) -> float:
        return self._derivatives[local_index]


class FastFiniteElement(FastMapping):
    """This element is made for better performance while calculating values and
    derivatives of Lagrange elements.

    Essentially it calculates finite elements on all points which are related to
    certain local points on the standard simplex. This is done by calculating
    the values (and if needed derivatives) in the preprocessing step by invoking
    'set_values' and 'set_derivatives'.

    The desired value can be accessed via 'value_on_simplex' or
    'derivative_on_simplex'.

    """

    _element_space: FiniteElementSpace
    _dof_vector: np.ndarray

    _local_basis_elements: List[FastLocalElement]
    _affine_transformation: AffineTransformation

    def __init__(self, element_space: FiniteElementSpace):
        self._element_space = element_space
        self._build_local_basis_elements()
        self._affine_transformation = AffineTransformation()

    def _build_local_basis_elements(self):
        self._local_basis_elements = [
            FastLocalElement(basis_element)
            for basis_element in self._element_space.local_basis
        ]

    @property
    def local_basis_elements(self) -> List[FastLocalElement]:
        return self._local_basis_elements

    def set_dof(self, dof_vector: np.ndarray):
        self._dof_vector = dof_vector

    def set_values(self, *local_points: float):
        for element in self._local_basis_elements:
            element.set_values(*local_points)

    def set_derivatives(self, *local_points: float):
        for element in self._local_basis_elements:
            element.set_derivatives(*local_points)

    def value(self, simplex_index: int, local_index: int) -> float:
        local_dof_vector = self._build_local_dof_vector(simplex_index)
        local_basis_elements_values = [
            element.value(local_index) for element in self._local_basis_elements
        ]
        return np.dot(local_dof_vector, local_basis_elements_values)

    def _build_local_dof_vector(self, simplex_index: int) -> np.ndarray:
        local_dof_vector = np.zeros(self._element_space.indices_per_simplex)

        for local_index in range(self._element_space.indices_per_simplex):
            global_index = self._element_space.get_global_index(
                simplex_index, local_index
            )
            local_dof_vector[local_index] = self._dof_vector[global_index]

        return local_dof_vector

    def derivative(self, simplex_index: int, local_index: int) -> float:
        local_dof_vector = self._build_local_dof_vector(simplex_index)
        local_basis_elements_derivatives = [
            element.derivative(local_index) for element in self._local_basis_elements
        ]

        simplex = self._element_space.mesh[simplex_index]
        distortion = self._affine_transformation.inverse_derivative(simplex)

        return distortion * np.dot(local_dof_vector, local_basis_elements_derivatives)


class QuadratureFastFiniteElement(FastFiniteElement):
    """Element for fast calculation of values and derivatives in quadrature
    nodes.

    """

    _quadrature_nodes: List[float]

    def __init__(
        self,
        element_space: FiniteElementSpace,
        local_quadrature: LocalElementQuadrature,
    ):
        FastFiniteElement.__init__(self, element_space)

        self._quadrature_nodes = local_quadrature.nodes

    def set_values(self):
        super().set_values(*self._quadrature_nodes)

    def set_derivatives(self):
        super().set_derivatives(*self._quadrature_nodes)


class AnchorNodesFastFiniteElement(FastFiniteElement):
    """Element for fast calculation of values and derivatives in anchor
    nodes of local Lagrange elements.

    """

    _anchor_nodes: List[float]

    def __init__(self, element_space: FiniteElementSpace):
        FastFiniteElement.__init__(self, element_space)

        self._anchor_nodes = element_space.local_basis.nodes

    def set_values(self):
        super().set_values(*self._anchor_nodes)

    def set_derivatives(self):
        super().set_derivatives(*self._anchor_nodes)
