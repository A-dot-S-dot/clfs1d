import numpy as np
from fem import FiniteElementSpace, LocalFiniteElement
from mesh import UniformMesh

from .assembler import LocalToGlobalMatrixAssembler
from .entry_calculator import ConstantSystemMatrixEntryCalculator
from .system_matrix import SystemMatrix


class DiscreteGradientEntryCalculator(ConstantSystemMatrixEntryCalculator):
    def __init__(self, element_space: FiniteElementSpace):
        ConstantSystemMatrixEntryCalculator.__init__(
            self, element_space, element_space.polynomial_degree
        )

    def _fill_local_matrix_entry(
        self, element_1: LocalFiniteElement, element_2: LocalFiniteElement
    ) -> float:
        # The transformation determinant and the derivative of affine
        # transformation cancel each other out

        quadrature_nodes_values = np.array(
            [
                element_1(node) * element_2.derivative(node)
                for node in self._local_quadrature.nodes
            ]
        )

        return np.dot(self._local_quadrature.weights, quadrature_nodes_values)


class DiscreteGradientMatrix(SystemMatrix):
    """Discrete gradient system matrix. It's entries are phi_i * phi'_j, where {phi_i}_i
    denotes the basis of the element space.

    """

    def __init__(self, element_space: FiniteElementSpace, build_inverse: bool = False):
        if not isinstance(element_space.mesh, UniformMesh):
            raise NotImplementedError(
                "Discrete gradient matrix can only be calculated for uniform meshes."
            )

        discrete_gradient_entry_calculator = DiscreteGradientEntryCalculator(
            element_space
        )
        assembler = LocalToGlobalMatrixAssembler(
            element_space, discrete_gradient_entry_calculator
        )

        SystemMatrix.__init__(self, element_space.dimension, assembler, build_inverse)
