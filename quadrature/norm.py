"""This module provides objects for calculating errors with finite elements."""
from abc import ABC, abstractmethod

import numpy as np
from math_types import FunctionRealToReal
from mesh import Simplex, UniformMesh
from quadrature.local import LocalElementQuadrature


class Norm(ABC):
    _mesh: UniformMesh

    def set_mesh(self, mesh: UniformMesh):
        self._mesh = mesh

    @abstractmethod
    def __call__(self, function: FunctionRealToReal) -> float:
        ...


class MeshDependentIntegralNorm(Norm):
    _local_quadrature: LocalElementQuadrature
    _mesh: UniformMesh
    _determinant_derivative_affine_mapping: float

    def __init__(self, mesh: UniformMesh, quadrature_degree: int):
        self._determinant_derivative_affine_mapping = mesh.step_length
        self._mesh = mesh
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)

    def set_mesh(self, mesh: UniformMesh):
        self._determinant_derivative_affine_mapping = mesh.step_length
        self._mesh = mesh


class L2Norm(MeshDependentIntegralNorm):
    def __call__(self, function: FunctionRealToReal) -> float:
        integral = 0

        for simplex in self._mesh:
            integral += self._calculate_norm_on_simplex(function, simplex)

        return np.sqrt(self._determinant_derivative_affine_mapping * integral)

    def _calculate_norm_on_simplex(
        self, function: FunctionRealToReal, simplex: Simplex
    ) -> float:
        node_values = np.array(
            [
                function(simplex.world_coordinates(node)) ** 2
                for node in self._local_quadrature.nodes
            ]
        )
        return np.dot(self._local_quadrature.weights, node_values)


class L1Norm(MeshDependentIntegralNorm):
    def __call__(self, function: FunctionRealToReal) -> float:
        integral = 0

        for simplex in self._mesh:
            integral += self._calculate_norm_on_simplex(function, simplex)

        return self._determinant_derivative_affine_mapping * integral

    def _calculate_norm_on_simplex(
        self, function: FunctionRealToReal, simplex: Simplex
    ) -> float:
        node_values = np.array(
            [
                np.absolute(function(simplex.world_coordinates(node)))
                for node in self._local_quadrature.nodes
            ]
        )
        return np.dot(self._local_quadrature.weights, node_values)


class LInfinityNorm(Norm):
    _mesh: UniformMesh
    _points_per_simplex: int

    def __init__(self, mesh: UniformMesh, points_per_simplex: int):
        self._mesh = mesh
        self._points_per_simplex = points_per_simplex

    def __call__(self, function: FunctionRealToReal) -> float:
        maximum_per_simplex = np.zeros(len(self._mesh))

        for simplex_index, simplex in enumerate(self._mesh):
            maximum_per_simplex[simplex_index] = self._calculate_norm_on_simplex(
                function, simplex
            )

        return np.amax(maximum_per_simplex)

    def _calculate_norm_on_simplex(
        self, function: FunctionRealToReal, simplex: Simplex
    ) -> float:
        return np.amax(
            [
                function(x)
                for x in np.linspace(simplex.a, simplex.b, self._points_per_simplex)
            ]
        )
