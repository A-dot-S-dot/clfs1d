"""This module provides different types for typing."""
from typing import Callable, Tuple

from numpy import ndarray

BarycentricCoordinate = Tuple[float, float]
RealD = ndarray

FunctionRealToReal = Callable[[float], float]
FunctionRealToRealD = Callable[[float], RealD]
FunctionRealDToReal = Callable[[RealD], float]
FunctionRealDToRealD = Callable[[RealD], RealD]
FunctionBarycentricToReal = Callable[[BarycentricCoordinate], float]
FunctionBarycentricToRealD = Callable[[BarycentricCoordinate], RealD]
