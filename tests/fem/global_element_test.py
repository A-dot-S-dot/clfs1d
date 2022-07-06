from unittest import TestCase

import numpy as np

from fem import GlobalFiniteElement

from ..test_helper import TestFiniteElementSpace


class TestLagragenFiniteElement(TestCase):
    element_space = TestFiniteElementSpace()
    element = GlobalFiniteElement(element_space, [2])
    points = np.linspace(0, 1)

    def test_not_dof_vector_error(self):
        self.assertRaises(ValueError, GlobalFiniteElement, self.element_space, (1, 2))

    def test_element_values(self):
        for point in self.points:
            self.assertEqual(self.element(point), 2)

    def test_element_derivative(self):
        for point in self.points:
            self.assertEqual(self.element.derivative(point), 0)
