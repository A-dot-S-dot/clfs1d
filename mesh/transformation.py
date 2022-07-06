"""This module provides Transformation between diffrent simplices."""
from typing import Tuple

import numpy as np
from math_types import BarycentricCoordinate

from .interval import Interval


class LocalCoordinates:
    """Mapping from a SIMPLEX to the Gibbs simplex."""

    _simplex: Interval
    _b_minus_a: float

    def __init__(self, simplex: Interval):
        self._simplex = simplex
        self._b_minus_a = simplex.b - simplex.a

    def __call__(self, x: float) -> BarycentricCoordinate:
        lambda_0 = (self._simplex.b - x) / self._b_minus_a
        lambda_1 = 1 - lambda_0

        return (lambda_0, lambda_1)

    @property
    def Lambda(self) -> np.ndarray:
        return np.array([-1 / self._b_minus_a, 1 / self._b_minus_a])


class WorldCoordinates:
    """Mapping from the Gibbs simplex to SIMPLEX."""

    _simplex: Interval

    def __init__(self, simplex: Interval):
        self._simplex = simplex

    def __call__(self, point: BarycentricCoordinate) -> float:
        return point[0] * self._simplex.a + point[1] * self._simplex.b


class AffineTransformation:
    """Mapping from standard simplex to SIMPLEX."""

    _simplex: Interval
    _b_minus_a: float

    def __init__(self, simplex: Interval):
        self._simplex = simplex
        self._b_minus_a = simplex.b - simplex.a

    def __call__(self, standard_simplex_point: float) -> float:
        return self._b_minus_a * standard_simplex_point + self._simplex.a

    def inverse(self, simplex_point: float) -> float:
        return (simplex_point - self._simplex.a) / self._b_minus_a

    @property
    def derivative(self) -> float:
        return self._b_minus_a
