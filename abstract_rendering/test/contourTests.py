from __future__ import print_function
import unittest
import numpy as np
from abstract_rendering.contour import Contour


class ContourTests(unittest.TestCase):
    pyramid2 = np.array(
            [[0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]])

    pyramid4 = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 2, 2, 2, 2, 1, 0],
             [0, 1, 2, 3, 3, 3, 1, 0],
             [0, 1, 2, 3, 3, 3, 1, 0],
             [0, 1, 2, 3, 3, 3, 1, 0],
             [0, 1, 2, 2, 2, 2, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]])

    def test_nlevels(self):
        self.assertListEqual(Contour.nlevels(self.pyramid2, 1).tolist(), [.5])
        self.assertListEqual(Contour.nlevels(self.pyramid2, 3).tolist(), [.25, .5, .75])
        self.assertListEqual(Contour.nlevels(self.pyramid4, 2).tolist(), [1,2])
        self.assertListEqual(Contour.nlevels(self.pyramid4, 3).tolist(), [.75, 1.5, 2.25])

    def test_fuse2(self):
        c = Contour(levels=1)
        contours = c.fuse(self.pyramid2)

        self.assertEquals(len(contours), 1)
        expected0 = [[(1.0, 0.5), 
                      (2.0, 0.5), 
                      (2.5, 1.0),
                      (2.5, 2.0),
                      (2.0, 2.5),
                      (1.0, 2.5),
                      (0.5, 2.0),
                      (0.5, 1.0),
                      (1.0, 0.5)]]

        self.assertListEqual(contours[.5], expected0)

    def test_fuse4(self):
        c = Contour(levels=3)
        levels = c.fuse(self.pyramid4)

        self.assertEquals(len(levels), 3)
        self.assertEquals(len(levels[.75][0]), 27)
        self.assertEquals(len(levels[1.5][0]), 19)
        self.assertEquals(len(levels[2.25][0]), 13)

        expected2 = [[(3.0, 2.25),
                      (4.0, 2.25),
                      (5.0, 2.25),
                      (5.375, 3.0),
                      (5.375, 4.0),
                      (5.375, 5.0),
                      (5.0, 5.75),
                      (4.0, 5.75),
                      (3.0, 5.75),
                      (2.25, 5.0),
                      (2.25, 4.0),
                      (2.25, 3.0),
                      (3.0, 2.25)]]

        self.assertListEqual(levels[2.25], expected2)

if __name__ == '__main__':
    unittest.main()
