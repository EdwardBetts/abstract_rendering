from __future__ import print_function

import unittest
import numpy as np
import abstract_rendering.numeric as numeric
import abstract_rendering.util as util
import abstract_rendering.core as core
from abstract_rendering.glyphset import ShapeCodes


class CountTests(unittest.TestCase):
    def test_allocate(self):
        op = numeric.Count()

        init = op.allocate(None, (10, 10))
        self.assertEquals(init.shape, (10, 10))
        self.assertTrue(np.array_equal(init, np.zeros((10, 10))))

    def test_combine(self):
        op = numeric.Count()

        glyph = [0, 0, 1, 1]
        existing = np.zeros((1, 1))
        op.combine(existing, glyph, ShapeCodes.POINT, 10)
        self.assertTrue(np.array_equal(existing, np.ones((1, 1))))

        existing = np.zeros((1, 1))
        op.combine(existing, glyph, ShapeCodes.POINT, 20)
        self.assertTrue(np.array_equal(existing, np.ones((1, 1))),
                        "Not input value invariant")

        existing = np.ones((1, 1))
        op.combine(existing, glyph, ShapeCodes.POINT, 20)
        self.assertTrue(np.array_equal(existing, np.array([[2]])),
                        "Count up from non-zero")

        glyph = [1, 2, 2, 3]
        existing = np.ones((3, 1))
        expected = np.array([[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 2, 1, 1]])
        op.combine(existing, glyph, ShapeCodes.POINT, 3)

        glyph = [5, 5, 6, 6]
        existing = np.ones((10, 10))
        expected = np.ones((10, 10))
        expected[5, 5] = 2
        op.combine(existing, glyph, ShapeCodes.POINT, 20)
        self.assertTrue(np.array_equal(existing, expected),
                        "Setting in the middle")

    def test_rollup(self):
        op = numeric.Count()

        ones = np.ones((5, 5))
        twos = np.empty((5, 5))
        twos.fill(2)

        result = op.rollup(ones, ones)
        self.assertTrue(np.array_equal(result, twos))


class SumTests(unittest.TestCase):
    def test_allocate(self):
        op = numeric.Sum()

        init = op.allocate(None, (10, 10))
        self.assertEquals(init.shape, (10, 10))
        self.assertTrue(np.array_equal(init, np.zeros((10, 10))))

    def test_combine(self):
        op = numeric.Sum()

        glyph = [0, 0, 1, 1]
        existing = np.ones((1, 1))
        expected = np.empty((1, 1))
        expected.fill(11)

        op.combine(existing, glyph, ShapeCodes.POINT, 10)
        self.assertTrue(np.array_equal(existing, expected))

        existing = np.ones((1, 1))
        expected = np.empty((1, 1))
        expected.fill(21)
        op.combine(existing, glyph, ShapeCodes.POINT, 20)
        self.assertTrue(np.array_equal(existing, expected))

        glyph = [1, 2, 2, 3]
        existing = np.ones((3, 1))
        expected = np.array([[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 4, 1, 1]])
        op.combine(existing, glyph, ShapeCodes.POINT, 3)

        glyph = [5, 5, 6, 6]
        existing = np.ones((10, 10))
        expected = np.ones((10, 10))
        expected[5, 5] = 21
        op.combine(existing, glyph, ShapeCodes.POINT, 20)
        self.assertTrue(np.array_equal(existing, expected),
                        "Setting in the middle")

    def test_rollup(self):
        op = numeric.Count()

        ones = np.ones((5, 5))
        twos = np.empty((5, 5))
        twos.fill(2)

        result = op.rollup(ones, ones)
        self.assertTrue(np.array_equal(result, twos))


def _test_extend(op1, tester):
    op2 = numeric.Cuberoot()
    tester.assertIsInstance(op1 + op2, core.Seq)
    tester.assertIsInstance(op2 + op1, core.Seq)


class FloorTests(unittest.TestCase):
    def test(self):
        op = numeric.Floor()

        a = np.array([[1.1, 1.5, 1.9],
                      [2.3, 0, 0.1]])

        expected = np.array([[1,   1,  1],
                             [2,   0,  0]], dtype=float)

        out = op.shade(a)
        self.assertTrue(np.array_equal(out, expected),
                        "Unequal:\n %s \n = \n %s" % (out, expected))

    def extend(self):
        _test_extend(numeric.Floor(), self)


class InterpolateTests(unittest.TestCase):
    def _run_test(self, low, high, msg):
        op = numeric.Interpolate(low, high)

        (width, height) = (5, 10)
        aggs = np.arange(0, width*height).reshape(height, width)
        out = op.shade(aggs)
        expected = np.linspace(low, high, width*height).reshape(height, width)
        self.assertTrue(np.allclose(out, expected), msg)

    def test_0to1(self):
        self._run_test(0, 1, "Zero to One")

    def test_0to10(self):
        self._run_test(0, 10, "Zero to Ten")

    def test_1to11(self):
        self._run_test(1, 11, "One to Eleven")

    def test_extend(self):
        _test_extend(numeric.Interpolate(1,2), self)


class PowerTests(unittest.TestCase):
    def test(self):
        op = numeric.Power(2)

        out = op.shade([0, 0, 0])
        self.assertTrue(np.array_equal(out, [0, 0, 0]))

        a = np.array([[1, 1, 1],
                      [2, 2, 2],
                      [5, 5, 5]])

        expected = np.array([[1,   1,  1],
                             [4,   4,  4],
                             [25, 25, 25]], dtype=float)

        out = op.shade(a)
        self.assertTrue(np.array_equal(out, expected),
                        "Unequal:\n %s \n = \n %s" % (out, expected))

    def test_extend(self):
        _test_extend(numeric.Power(3), self)


class CuberootTests(unittest.TestCase):
    def test(self):
        op = numeric.Cuberoot()

        out = op.shade([1, 1, 1])
        self.assertTrue(np.array_equal(out, [1, 1, 1]))

        a = np.array([[27],
                      [64],
                      [125]])

        # Calculates in-place because of floating point round-off
        expected = np.array([[pow(27, 1/3.0)],
                             [pow(64, 1/3.0)],
                             [pow(125, 1/3.0)]], dtype=float)

        out = op.shade(a)
        self.assertTrue(np.array_equal(out, expected),
                        "Unequal:\n %s \n = \n %s" % (out, expected))

    def test_extend(self):
        _test_extend(numeric.Cuberoot(), self)


class SqrtTests(unittest.TestCase):
    def test(self):
        op = numeric.Sqrt()

        out = op.shade([1, 1, 1])
        self.assertTrue(np.array_equal(out, [1, 1, 1]))

        a = np.array([[4,   4,  4],
                      [9,   9,  9],
                      [25, 25, 25]])

        expected = np.array([[2, 2, 2],
                             [3, 3, 3],
                             [5, 5, 5]], dtype=float)

        out = op.shade(a)
        self.assertTrue(np.array_equal(out, expected),
                        "Unequal:\n %s \n = \n %s" % (out, expected))

    def test_extend(self):
        _test_extend(numeric.Sqrt(), self)


class BinarySegmentTests(unittest.TestCase):
    def test(self):
        w = util.Color(255, 255, 255, 255)
        b = util.Color(0, 0, 0, 255)
        op = numeric.BinarySegment(b, w, 5)

        aggs = np.array([[0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1],
                         [5, 5, 5, 5, 5],
                         [6, 6, 6, 6, 6],
                         [9, 9, 9, 9, 9],
                         [0, 1, 5, 6, -1]])
        out = op.shade(aggs)
        expected = np.array([[b, b, b, b, b],
                             [b, b, b, b, b],
                             [w, w, w, w, w],
                             [w, w, w, w, w],
                             [w, w, w, w, w],
                             [b, b, w, w, b]])
        self.assertTrue(np.array_equal(out, expected))

    def test_extend(self):
        _test_extend(numeric.BinarySegment(None, None, 3), self)


class InterpolateColorsTests(unittest.TestCase):
    def test_linear(self):
        red = util.Color(255, 0, 0, 255)
        white = util.Color(255, 255, 255, 255)
        op = numeric.InterpolateColors(white, red)

        aggs = np.arange(0, 256).reshape((32, 8))
        out = op.shade(aggs)

        var = np.arange(0, 256)[::-1].reshape((32, 8))
        const = np.empty((32, 8), dtype=np.int)
        const.fill(255)
        expected = np.dstack((const, var, var, const))
        self.assertTrue(np.array_equal(out, expected))

    def test_extend(self):
        _test_extend(numeric.InterpolateColors(None, None), self)


class SpreadTests(unittest.TestCase):
    def run_spread(self, spread, in_vals, expected):
        op = numeric.Spread(factor=spread)
        out = op.shade(in_vals)

        self.assertTrue(np.array_equal(out, expected),
                        'incorrect value spreading\ni %s' % str(out))

    def test_spread_oneseed(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]])
        self.run_spread(1, a, ex)

    def test_spread_twoseeds(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[0, 0, 0, 0, 0, 0],
                         [0, 1, 2, 2, 1, 0],
                         [0, 1, 2, 2, 1, 0],
                         [0, 1, 2, 2, 1, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]])
        self.run_spread(1, a, ex)

    def test_spread_two(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]])
        self.run_spread(2, a, ex)

    def test_spread_zero(self):
        a = np.asarray([[0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        ex = np.asarray([[0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
        self.run_spread(0, a, ex)

    def test_extend(self):
        _test_extend(numeric.Spread(), self)


if __name__ == '__main__':
    unittest.main()
