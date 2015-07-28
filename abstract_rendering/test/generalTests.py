from __future__ import print_function

import unittest
import numpy as np
import abstract_rendering.general as general


class IdTests(unittest.TestCase):
    def test(self):
        op = general.Id()

        aggs = np.zeros((4, 3))
        out = op.shade(aggs)
        self.assertIs(aggs, out)

if __name__ == '__main__':
    unittest.main()
