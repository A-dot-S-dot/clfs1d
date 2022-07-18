from math_types import FunctionRealDToRealD
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import SuperLU, splu, spsolve

from .assembler import MatrixAssembler


class SystemMatrix(csc_matrix):
    _inverse: SuperLU
    _inverse_function: FunctionRealDToRealD

    def __init__(
        self,
        element_space_dimension: int,
        assembler: MatrixAssembler,
        build_inverse: bool = False,
    ):
        matrix = lil_matrix((element_space_dimension, element_space_dimension))
        assembler.fill_entries(matrix)

        csc_matrix.__init__(self, matrix)

        if build_inverse:
            self._build_inverse()
        else:
            self._inverse_function = lambda vector: spsolve(self, vector)

    def _build_inverse(self):
        self._inverse = splu(self)
        self._inverse_function = lambda vector: self._inverse.solve(vector)

    @property
    def inverse(self) -> FunctionRealDToRealD:
        return self._inverse_function

    def __str__(self) -> str:
        return self.toarray().__str__()
