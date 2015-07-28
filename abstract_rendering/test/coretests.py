from __future__ import print_function

import unittest
import abstract_rendering.core as core
import abstract_rendering.numeric as numeric
import numpy as np

class Seq(unittest.TestCase):
    def test_extend(self):
        op1 = numeric.Cuberoot()
        op2 = numeric.Cuberoot()
        op3 = numeric.Cuberoot()

        seq1 = core.Seq(op1, op2)
        seq2 = seq1+op3

        self.assertIsInstance(seq2, core.Seq)
        self.assertIsNot(seq1, seq2)

    def test_call(self):
        op1 = numeric.Cuberoot()
        op2 = numeric.Cuberoot()
        op3 = op1 + op2

        rslt = op3(np.array([[1]]))
        self.assertEquals(rslt, [[1]])

        start = [[(3**3)**3]]
        rslt = op3(np.array(start))
        self.assertEquals(rslt, op2(op1(start)))


if __name__ == '__main__':
    unittest.main()
