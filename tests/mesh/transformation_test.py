from unittest import TestCase

import numpy as np
from ..test_helper import discrete_derivative

from mesh import Simplex
from mesh.transformation import AffineTransformation, LocalCoordinates, WorldCoordinates


class TestLocalAndWorldCoordiantes(TestCase):
    simplex = Simplex(0, 1)
    simplex_points = [0, 0.5, 1]
    barycentric_points = [(1, 0), (0.5, 0.5), (0, 1)]
    local_coordinates = LocalCoordinates(simplex)
    world_coordinates = WorldCoordinates(simplex)

    def test_local_coordinates(self):
        for x, (lambda_0, lambda_1) in zip(
            self.simplex_points, self.barycentric_points
        ):
            self.assertTupleEqual(self.local_coordinates(x), (lambda_0, lambda_1))

    def test_world_coordinates(self):
        for x, barycentric_point in zip(self.simplex_points, self.barycentric_points):
            self.assertEqual(self.world_coordinates(barycentric_point), x)

    def test_inverse_property(self):

        for x in self.simplex_points:
            self.assertEqual(x, self.world_coordinates(self.local_coordinates(x)))

        for barycentric_point in self.barycentric_points:
            self.assertTupleEqual(
                barycentric_point,
                self.local_coordinates(self.world_coordinates(barycentric_point)),
            )

    def test_Lambda(self):
        Lambda = self.local_coordinates.Lambda
        for x in np.linspace(self.simplex.a, self.simplex.b):
            self.assertAlmostEqual(
                Lambda[0],
                discrete_derivative(lambda x: self.local_coordinates(x)[0], x),
            )
            self.assertAlmostEqual(
                Lambda[1],
                discrete_derivative(lambda x: self.local_coordinates(x)[1], x),
            )


class TestAffineTransformation(TestCase):
    simplex = Simplex(-1, 1)
    simplex_points = [-1, 0, 1]
    standard_simplex_points = [0, 0.5, 1]
    affine_transformation = AffineTransformation(simplex)

    def test_call(self):
        for x, x_standard in zip(self.simplex_points, self.standard_simplex_points):
            self.assertEqual(self.affine_transformation(x_standard), x)

    def test_inverse(self):
        for x, x_standard in zip(self.simplex_points, self.standard_simplex_points):
            self.assertEqual(self.affine_transformation.inverse(x), x_standard)

    def test_inverse_property(self):
        for x, x_standard in zip(self.simplex_points, self.standard_simplex_points):
            self.assertEqual(
                self.affine_transformation.inverse(self.affine_transformation(x)), x
            )
            self.assertEqual(
                self.affine_transformation(
                    self.affine_transformation.inverse(x_standard)
                ),
                x_standard,
            )

    def test_derivative(self):
        for x in np.linspace(self.simplex.a, self.simplex.b):
            self.assertAlmostEqual(
                self.affine_transformation.derivative,
                discrete_derivative(self.affine_transformation, x),
            )
