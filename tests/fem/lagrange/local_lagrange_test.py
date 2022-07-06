from unittest import TestCase

import numpy as np
from fem.lagrange.local_lagrange import LocalLagrangeBasis

from ...test_helper import discrete_derivative


class TestLocalLagrangeBasis(TestCase):
    test_polynomial_degrees = [p + 1 for p in range(8)]
    local_bases = [LocalLagrangeBasis(p) for p in test_polynomial_degrees]

    def test_nodes(self):
        expected_nodes = [
            [(1, 0), (0, 1)],
            [(1, 0), (0.5, 0.5), (0, 1)],
            [(1, 0), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (0, 1)],
        ]

        for local_basis, expected_basis_nodes in zip(
            self.local_bases[:3], expected_nodes
        ):
            for node, expected_node in zip(local_basis.nodes, expected_basis_nodes):
                self.assertTupleEqual(node, expected_node)

    def test_delta_property(self):
        for local_basis in self.local_bases:
            for i, node_1 in enumerate(local_basis.nodes):
                basis_element = local_basis.get_element_at_node(node_1)
                for j, node_2 in enumerate(local_basis.nodes):
                    self.assertAlmostEqual(
                        basis_element(node_2),
                        float(node_1 == node_2),
                        msg=f"p={local_basis.polynomial_degree}, basis_index={i}, node_index={j}",
                    )

    def test_len(self):
        for local_basis in self.local_bases:
            self.assertEqual(len(local_basis), local_basis.polynomial_degree + 1)

    def test_derivative(self):
        for local_basis in self.local_bases[:4]:
            for element_index, element in enumerate(local_basis):
                for lambda_0 in np.linspace(0, 1):
                    lambda_1 = 1 - lambda_0
                    self.assertAlmostEqual(
                        element.derivative((lambda_0, lambda_1))[0],
                        discrete_derivative(lambda x: element((x, lambda_1)), lambda_0),
                        msg=f"entry=0, p={local_basis.polynomial_degree} , element={element_index}, point=({lambda_0},{lambda_1})",
                        delta=1e-7,
                    )
                    self.assertAlmostEqual(
                        element.derivative((lambda_0, lambda_1))[1],
                        0,
                        msg=f"entry=1, p={local_basis.polynomial_degree}, element={element_index}, point=({lambda_0},{lambda_1})",
                    )
