from __future__ import print_function

import abstract_rendering.geometry as geometry
import numpy as np
import unittest


class BressenhamTests(unittest.TestCase):
    def test_vertical(self):
        out = np.zeros((10, 1))
        geometry.bressenham(out, [0, 0, 0, 9], 1)
        expected = np.ones((10, 1))
        self.assertTrue(np.array_equal(out, expected), "Simple vertical")

        geometry.bressenham(out, [0, 9, 0, 0], 1)
        expected = np.ones((10, 1))
        self.assertTrue(np.array_equal(out, expected), "Reverse vertical")

        expected.fill(7)
        geometry.bressenham(out, [0, 0, 0, 9], 7)
        self.assertTrue(np.array_equal(out, expected), "Vertical: Another value")

        out.fill(0)
        expected.fill(7)
        expected[0, 0] = 0
        expected[9, 0] = 0
        geometry.bressenham(out, [0, 1, 0, 8], 7)
        self.assertTrue(np.array_equal(out, expected), "Vertical: Not space filling")

    def test_horizontal(self):
        out = np.zeros((1, 10))
        geometry.bressenham(out, [0, 0, 9, 0], 1)
        expected = np.ones((1, 10))
        self.assertTrue(np.array_equal(out, expected), "Simple horizontal")

        geometry.bressenham(out, [9, 0, 0, 0], 1)
        expected = np.ones((1, 10))
        self.assertTrue(np.array_equal(out, expected), "Reverse horizontal")

        expected.fill(7)
        geometry.bressenham(out, [0, 0, 9, 0], 7)
        self.assertTrue(np.array_equal(out, expected), "Horizontal: Another value")

        out.fill(0)
        expected.fill(7)
        expected[0, 0] = 0
        expected[0, 9] = 0
        geometry.bressenham(out, [1, 0, 8, 0], 7)
        self.assertTrue(np.array_equal(out, expected), "Horizontal: Not space filling")

    def test_diag(self):
        out = np.zeros((10, 11), dtype=np.int)
        geometry.bressenham(out, [2, 2, 6, 5], 1)
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.array_equal(out, expected))

        out = np.zeros((10, 11), dtype=np.int)
        geometry.bressenham(out, [7, 3, 1, 8], 1)
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.array_equal(out, expected))

    def test_compound(self):
        cats = [1, 2]
        out = np.zeros((5, 1, len(cats)))
        geometry.bressenham(out, [0, 0, 0, 4], cats)
        expected = np.empty(out.shape)
        expected[:, :, 0] = 1
        expected[:, :, 1] = 2
        self.assertTrue(np.array_equal(out, expected), "Complex value")


if __name__ == '__main__':
    unittest.main()
