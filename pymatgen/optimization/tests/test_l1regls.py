# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import unicode_literals

import unittest

from pymatgen.optimization.l1regls import l1regls
import numpy as np


class L1regTest(unittest.TestCase):

    def test(self):
        A1 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
        b = np.array([1, 2, 3])
        x1 = l1regls(A1, b)
        res1 = np.array([0.5, 1.5, 2.5])
        self.assertTrue(np.allclose(x1, res1))

        x2 = l1regls(A1, b, 2)
        res2 = np.array([0.75, 1.75, 2.75])
        self.assertTrue(np.allclose(x2, res2))


if __name__ == "__main__":
    unittest.main()
