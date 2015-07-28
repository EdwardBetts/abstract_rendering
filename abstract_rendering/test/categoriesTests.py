from __future__ import print_function
from six.moves import reduce 

import unittest
import abstract_rendering.categories as categories
from abstract_rendering.glyphset import ShapeCodes
from abstract_rendering.util import Color
import numpy as np
import operator

class _GlyphsetShim(object):
    def __init__(self, depth):
        self.depth = depth

    def data(self):
        return range(self.depth)

class CountCategories(unittest.TestCase):
    def test_allocate(self):
        op = categories.CountCategories()
        (width, height, depth) = (6, 3, 7)
        screen = (width, height)
        glyphset = _GlyphsetShim(depth)
        out = op.allocate(glyphset, screen)

        expected = np.zeros((height, width, depth))
        self.assertTrue(np.array_equal(out, expected))

    def test_combine_points(self):
        op = categories.CountCategories()

        (width, height, depth) = (1, 1, 3)
        screen = (width, height)
        glyphset = _GlyphsetShim(depth)

        out = op.allocate(glyphset, screen)
        glyph = [0, 0, 1, 1]
        existing = op.allocate(glyphset, screen)
        op.combine(existing, glyph, ShapeCodes.POINT, 1)

        expected = np.zeros((height, width, depth))
        expected[0, 0, 1] = 1
        self.assertTrue(np.array_equal(existing, expected))

        (width, height, depth) = (3, 4, 3)
        screen = (width, height)
        glyphset = _GlyphsetShim(depth)
        glyph = [2, 1, 3, 2]
        existing = op.allocate(glyphset, screen)
        op.combine(existing, glyph, ShapeCodes.POINT, 2)

        expected = np.zeros((height, width, depth), dtype=np.int)
        expected[1, 2, 2] = 1
        self.assertTrue(np.array_equal(existing, expected))

    def test_combine_rect(self):
        op = categories.CountCategories()
        (width, height, depth) = (3, 4, 2)
        screen = (width, height)
        glyphset = _GlyphsetShim(depth)

        glyph = [0, 0, 2, 2]
        existing = op.allocate(glyphset, screen)
        op.combine(existing, glyph, ShapeCodes.RECT, 0)

        expected = np.array([[[1, 1, 0],
                              [1, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]], dtype=np.int).transpose((1, 2, 0))

        self.assertEqual(existing.shape, expected.shape)
        self.assertTrue(np.array_equal(existing, expected))

    def test_rollup(self):
        op = categories.CountCategories()
        (width, height, depth) = (8, 5, 3)

        ones = np.ones((depth, height, width))
        zeros = np.zeros((depth, height, width))
        twos = np.empty((depth, height, width))
        twos.fill(2)
        threes = np.empty((depth, height, width))
        threes.fill(3)

        out = op.rollup(ones, zeros)
        self.assertTrue(np.array_equal(out, ones))

        out = op.rollup(ones, ones)
        self.assertTrue(np.array_equal(out, twos))

        out = op.rollup(twos, ones)
        self.assertTrue(np.array_equal(out, threes))


class ToCounts(unittest.TestCase):
    def test(self):
        """Can the output of categories.CountCategories can be
        transformed to vanilla counts?"""

        op = categories.ToCounts()
        (width, height, depth) = (4, 5, 3)
        screen = (width, height)
        glyphset = _GlyphsetShim(depth)

        aggregator = categories.CountCategories()
        aggs = aggregator.allocate(glyphset, screen)
        aggs.fill(1)
        out = op.shade(aggs)
        expected = np.empty((height, width), dtype=np.int32)
        expected.fill(depth)
        self.assertTrue(np.array_equal(out, expected))
        self.assertEquals((height, width), out.shape, "Unexpected out shape")

        shape = aggs.shape
        aggs = np.arange(0, reduce(operator.mul, shape))
        aggs = aggs.reshape((depth, height, width)).transpose((1, 2, 0))
        out = op.shade(aggs)

        self.assertEquals((height, width), out.shape, "Unexpected out shape")

        self.assertEquals(out[0, 0], 0+20+40)
        self.assertEquals(out[4, 0], 16+36+56)
        self.assertEquals(out[0, 3], 3+23+43)
        self.assertEquals(out[4, 3], 19+39+59)


class Select(unittest.TestCase):
    def test(self):
        op0 = categories.Select(0)
        op1 = categories.Select(1)
        op2 = categories.Select(2)
        (width, height, depth) = (5, 4, 3)
        screen = (width, height)
        glyphset = _GlyphsetShim(depth)

        aggregator = categories.CountCategories()
        aggs = aggregator.allocate(glyphset, screen)
        aggs[:, :, 0] = 1
        aggs[:, :, 1] = 2
        aggs[:, :, 2] = 3

        expected = np.empty((height, width, depth), dtype=np.int)
        expected[:, :, 0] = 1
        expected[:, :, 1] = 2
        expected[:, :, 2] = 3

        self.assertTrue(np.array_equal(op0.shade(aggs), expected[:, :, 0]))
        self.assertTrue(np.array_equal(op1.shade(aggs), expected[:, :, 1]))
        self.assertTrue(np.array_equal(op2.shade(aggs), expected[:, :, 2]))


class MinPercent(unittest.TestCase):
    def test_uniform(self):
        op5 = categories.MinPercent(.5)
        op1 = categories.MinPercent(.1)

        (width, height, depth) = (4, 6, 4)
        screen = (width, height)
        glyphset = _GlyphsetShim(depth)

        aggregator = categories.CountCategories()
        aggs = aggregator.allocate(glyphset, screen)
        aggs.fill(2)

        above = np.empty((height, width, 4), dtype=np.uint8)
        above[:] = op5.above
        out = op1.shade(aggs)
        self.assertTrue(np.array_equal(out, above))

        below = np.empty((height, width, 4), dtype=np.uint8)
        below[:] = op5.below
        out = op5.shade(aggs)
        self.assertTrue(np.array_equal(out, below))

    def test_mixed(self):
        op0_5 = categories.MinPercent(.5)
        op0_25 = categories.MinPercent(.25)
        op1_5 = categories.MinPercent(.5, cat=1)
        op2_25 = categories.MinPercent(.25, cat=2)

        aggs = np.array([[[0, 1, 2, 3, 4]],
                         [[1, 2, 3, 4, 0]],
                         [[2, 3, 4, 0, 1]],
                         [[3, 4, 0, 1, 2]]]).transpose((1, 2, 0))

        above = op0_5.above
        below = op0_5.below
        expected0_5 = np.array([[below, below, below, below, above]])
        expected0_25 = np.array([[below, below, below, above, above]])
        expected1_5 = np.array([[below, below, below, above, below]])
        expected2_25 = np.array([[above, above, above, below, below]])

        self.assertTrue(np.array_equal(op0_5.shade(aggs), expected0_5))
        self.assertTrue(np.array_equal(op0_25.shade(aggs), expected0_25))
        self.assertTrue(np.array_equal(op1_5.shade(aggs), expected1_5))
        self.assertTrue(np.array_equal(op2_25.shade(aggs), expected2_25))


class HDAlpha(unittest.TestCase):
    red = Color(255, 0, 0, 255)
    green = Color(0, 255, 0, 255)
    blue = Color(0, 0, 255, 255)

    def test_empty(self):
        op = categories.HDAlpha([self.red, self.green, self.blue])

        aggs = np.zeros((3, 3, 3))
        expected = np.empty((3, 3, 4), dtype=np.int)
        expected[:] = op.background
        self.assertTrue(np.array_equal(op.shade(aggs), expected))

    def test_simple(self):
        op = categories.HDAlpha([self.red, self.green, self.blue])

        aggs = np.array([[[1, 0, 0, 0]],
                         [[0, 1, 0, 0]],
                         [[0, 0, 1, 0]]]).transpose((1, 2, 0))
        expected = np.array([[self.red, self.green, self.blue, op.background]])
        self.assertTrue(np.array_equal(op.shade(aggs), expected))

    def test_blend(self):
        op = categories.HDAlpha([self.red, self.green, self.blue])

        aggs = np.array([[[1, 0, 0]],
                         [[1, 1, 0]],
                         [[0, 1, 1]]]).transpose((1, 2, 0))

        expected = np.array([[[127, 127,   0, 255],
                              [  0, 127, 127, 255],
                              [  0,   0, 255, 127]]], dtype=np.uint8)

        self.assertTrue(np.array_equal(op.shade(aggs), expected))


if __name__ == '__main__':
    unittest.main()
