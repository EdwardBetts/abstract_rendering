from __future__ import print_function

import unittest
import abstract_rendering.util as util


class EmptyListTests(unittest.TestCase):
    def test_without_length(self):
        ls = util.EmptyList()
        self.assertIsNone(ls[0])
        self.assertIsNone(ls[-1])
        self.assertIsNone(ls[30])
        self.assertIsNone(ls[9048])
        self.assertIsNone(ls[100398384])
        self.assertIsNone(ls[3])
        self.assertIsNone(ls[490])

    def test_with_length(self):
        ls = util.EmptyList(7)
        self.assertIsNone(ls[0])
        self.assertIsNone(ls[1])
        self.assertIsNone(ls[2])
        self.assertIsNone(ls[3])
        self.assertIsNone(ls[4])
        self.assertIsNone(ls[5])
        self.assertIsNone(ls[6])

        self.assertRaises(IndexError, ls.__getitem__, -1)
        self.assertRaises(IndexError, ls.__getitem__, 8)


class ColorTest(unittest.TestCase):
    def _test(self, r, g, b, a):
        c = util.Color(r, g, b, a)
        self.assertEqual(util.Color(r, g, b, a), c)
        self.assertEqual(c,  [r, g, b, a])

    def test1(self): self._test(0, 0, 0, 0)
    def test2(self): self._test(10, 30, 40, 20)
    def test3(self): self._test(255, 255, 255, 255)

    def test_RedError(self):
        self.assertRaises(ValueError, util.Color, 256, 0, 0, 0)
        self.assertRaises(ValueError, util.Color, -1, 0, 0, 0)

    def test_GreenError(self):
        self.assertRaises(ValueError, util.Color, 0, 256, 0, 0)
        self.assertRaises(ValueError, util.Color, 0, -1, 0, 0)

    def test_BlueError(self):
        self.assertRaises(ValueError, util.Color, 0, 0, 256, 0)
        self.assertRaises(ValueError, util.Color, 0, 0, -1, 0)

    def test_AlphaError(self):
        self.assertRaises(ValueError, util.Color, 0, 0, 0, 256)
        self.assertRaises(ValueError, util.Color, 0, 0, 0, -1)


class ZoomFitTest(unittest.TestCase):
    def test_scale(self):
        self.assertEqual(util.zoom_fit((10, 10), (0, 0, 10, 10)),
                         [0., 0., 1., 1.])

        self.assertEqual(util.zoom_fit((10, 10), (0, 0, 20, 20)),
                         [0., 0., .5, .5])

        self.assertEqual(util.zoom_fit((10, 10), (0, 0, 5, 5)),
                         [0., 0., 2., 2.])

        self.assertEqual(util.zoom_fit((10, 10), (0, 0, 10, 20)),
                         [0., 0., .5, .5])

        self.assertEqual(util.zoom_fit((10, 10), (0, 0, 10, 20), False),
                         [0., 0., 1., .5])

    def test_pan(self):
        self.assertEqual(util.zoom_fit((10, 10), (0, 0, 10, 10)),
                         [0., 0., 1., 1.])

        self.assertEqual(util.zoom_fit((10, 10), (5, 5, 10, 10)),
                         [-5., -5., 1., 1.])

        self.assertEqual(util.zoom_fit((10, 10), (-4, -7, 10, 10)),
                         [4., 7., 1., 1.])

    def test_pan_scale(self):
        self.assertEqual(util.zoom_fit((10, 10), (0, 0, 20, 20)),
                         [0., 0., .5, .5])

        self.assertEqual(util.zoom_fit((10, 10), (5, 5, 20, 20)),
                         [-2.5, -2.5, .5, .5])

        self.assertEqual(util.zoom_fit((10, 10), (-4, -7, 20, 20)),
                         [2., 3.5, .5, .5])


if __name__ == '__main__':
    unittest.main()
