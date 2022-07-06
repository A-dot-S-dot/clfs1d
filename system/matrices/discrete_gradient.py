import numpy as np
from fem import FiniteElementSpace, LocalFiniteElement
from mesh import UniformMesh

from .assembler import LocalToGlobalMatrixAssembler
from .entry_calculator import ConstantSystemMatrixEntryCalculator
from .system_matrix import SystemMatrix


class DiscreteGradientEntryCalculator(ConstantSystemMatrixEntryCalculator):
    _Lambda: np.ndarray
    _determinant_derivative_affine_mapping: float

    def __init__(self, element_space: FiniteElementSpace):
        if not isinstance(element_space.mesh, UniformMesh):
            raise NotImplementedError(
                "Mass matrix can only be calculated for uniform meshes."
            )

        step_length = element_space.mesh.step_length
        self._determinant_derivative_affine_mapping = step_length
        self._Lambda = np.array([-1 / step_length, 1 / step_length])

        ConstantSystemMatrixEntryCalculator.__init__(
            self, element_space, element_space.polynomial_degree
        )

    def _fill_local_matrix_entry(
        self, element_1: LocalFiniteElement, element_2: LocalFiniteElement
    ) -> float:
        quadrature_nodes_values = np.array(
            [
                element_1(node) * np.dot(self._Lambda, element_2.derivative(node))
                for node in self._local_quadrature.nodes
            ]
        )

        return self._determinant_derivative_affine_mapping * np.dot(
            self._local_quadrature.weights, quadrature_nodes_values
        )


class DiscreteGradientMatrix(SystemMatrix):
    """Mass system matrix. It's entries are phi_i * phi'_j, where {phi_i}_i denotes
    the basis of the element space."""

    def __init__(self, element_space: FiniteElementSpace, build_inverse: bool = False):
        discrete_gradient_entry_calculator = DiscreteGradientEntryCalculator(
            element_space
        )
        assembler = LocalToGlobalMatrixAssembler(
            element_space, discrete_gradient_entry_calculator
        )

        SystemMatrix.__init__(self, element_space.dimension, assembler, build_inverse)
