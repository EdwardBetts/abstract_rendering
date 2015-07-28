from __future__ import print_function

import unittest
import numpy as np
import abstract_rendering.numpyglyphs as npg


class SpreadTests(unittest.TestCase):
    def run_spread(self, spread, in_vals, expected, **kwargs):
        op = npg.Spread(factor=spread, **kwargs)
        out = op.shade(in_vals)

        self.assertTrue(np.array_equal(out, expected),
                        'incorrect value spreading\n %s' % str(out))

    def test_spread_oneseed(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[1, 1, 1, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 0],
                         [0, 0, 0, 0]])
        self.run_spread(2, a, ex, shape="rect")

    def test_spread_twoseeds(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[1, 2, 2, 1],
                         [1, 2, 2, 1],
                         [1, 2, 2, 1],
                         [0, 0, 0, 0]])
        self.run_spread(2, a, ex, shape="rect")



    def test_spread_two_anti_alias(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.array([[0,   0, 0, 0],
                       [0,   0, 0, 0],
                       [.5, .5, 0, 0],
                       [.5, .5, 0, 0],
                       [0,   0, 0, 0],
                       [0,   0, 0, 0],
                       [0,   0, 0, 0],
                       [0,   0, 0, 0]], dtype=np.float64)
        self.run_spread(2, a, ex, anti_alias=True, shape="rect")

    def test_spread_two(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
        self.run_spread(2, a, ex, shape="rect")

    def test_spread_three(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[0, 0, 0, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
        self.run_spread(3, a, ex, shape="rect")

    def test_spread_four(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1]])
        self.run_spread(4, a, ex, shape="rect")

    def test_spread_zero(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
        self.run_spread(0, a, ex, shape="rect")


    def test_spread_cats(self):
        a = np.asarray([[[0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]],
                        [[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 0]]]).T
        
        b = np.asarray([[[1, 1, 1, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 0],
                         [0, 0, 0, 0]],
                        [[0, 0, 0, 0],
                         [0, 1, 1, 1],
                         [0, 1, 1, 1],
                         [0, 1, 1, 1]]]).T
        self.run_spread(2, a, b, shape="rect")


    def test_spread_circle2(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [1, 1, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
        self.run_spread(2, a, ex, shape="circle")

    def test_spread_circle4(self):
        a = np.asarray([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])

        ex = np.asarray([[0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])
        self.run_spread(4, a, ex, shape="circle")

if __name__ == '__main__':
    unittest.main()
