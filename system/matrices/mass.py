"""Provides mass system matrix."""
import numpy as np
from fem import FiniteElementSpace, LocalFiniteElement
from mesh import UniformMesh

from .assembler import LocalToGlobalMatrixAssembler
from .entry_calculator import ConstantSystemMatrixEntryCalculator
from .system_matrix import SystemMatrix


class MassEntryCalculator(ConstantSystemMatrixEntryCalculator):
    _determinant_derivative_affine_mapping: float

    def __init__(self, element_space: FiniteElementSpace):
        self._determinant_derivative_affine_mapping = element_space.mesh.step_length
        ConstantSystemMatrixEntryCalculator.__init__(
            self, element_space, element_space.polynomial_degree + 1
        )

    def _fill_local_matrix_entry(
        self, element_1: LocalFiniteElement, element_2: LocalFiniteElement
    ) -> float:
        quadrature_nodes_weights = np.array(
            [element_1(node) * element_2(node) for node in self._local_quadrature.nodes]
        )

        return self._determinant_derivative_affine_mapping * np.dot(
            self._local_quadrature.weights, quadrature_nodes_weights
        )


class MassMatrix(SystemMatrix):
    """Mass system matrix. It's entries are phi_i * phi_j, where {phi_i}_i denotes
    the basis of the element space."""

    def __init__(self, element_space: FiniteElementSpace, build_inverse: bool = False):

        if not isinstance(element_space.mesh, UniformMesh):
            raise NotImplementedError(
                "Mass matrix can only be calculated for uniform meshes."
            )

        mass_entry_calculator = MassEntryCalculator(element_space)
        assembler = LocalToGlobalMatrixAssembler(element_space, mass_entry_calculator)

        SystemMatrix.__init__(self, element_space.dimension, assembler, build_inverse)
