"""Provides assembling classes of VMS stabilization vector.

The VMS stabilization is defined via

    s(uh, vi) = c*I(Dvi*(Duh-gh), Omega),

where c=h/2p and gh is a gradient approximation.

See 'Entropy conservation property and entropy stabilization of high-order
continuous Galerkin approximations to scalar conservation laws', D. Kuzmin, M.
Quezada de Luna, 2020, p. 3
"""

import numpy as np
from fem import FiniteElementSpace

from .builder import DOFVectorBuilder
from .l2_product_correction_basis import (
    L2ProductCorrectionBasisBuilder,
    L2ProductCorrectionDerivativeBasisEntryCalculator,
)


class VMSStabilizationBuilder(DOFVectorBuilder):
    _element_space: FiniteElementSpace
    _stabilization_factor: float
    _gradient_approximation_builder: DOFVectorBuilder
    _l2_product_correction_derivative_basis_builder: L2ProductCorrectionBasisBuilder

    def __init__(
        self,
        element_space: FiniteElementSpace,
        stabilization_parameter: float,
        gradient_approximation_builder: DOFVectorBuilder,
    ):
        self._element_space = element_space
        self._build_stabilization_factor(stabilization_parameter)
        self._gradient_approximation_builder = gradient_approximation_builder
        self._build_l2_product_builder()

    def _build_stabilization_factor(self, stabilization_parameter: float):
        self._stabilization_factor = (
            stabilization_parameter
            * self._element_space.mesh.step_length
            / (2 * self._element_space.polynomial_degree)
        )

    def _build_l2_product_builder(self):
        entry_calculator = L2ProductCorrectionDerivativeBasisEntryCalculator(
            self._element_space,
            self._element_space.polynomial_degree,
        )
        self._l2_product_correction_derivative_basis_builder = (
            L2ProductCorrectionBasisBuilder(self._element_space, entry_calculator)
        )

    def build_vector(self, discrete_solution_dof_vector: np.ndarray) -> np.ndarray:
        gradient_approximation_dof_vector = (
            self._gradient_approximation_builder.build_vector(
                discrete_solution_dof_vector
            )
        )
        return (
            self._stabilization_factor
            * self._l2_product_correction_derivative_basis_builder.build_vector(
                discrete_solution_dof_vector, gradient_approximation_dof_vector
            )
        )
