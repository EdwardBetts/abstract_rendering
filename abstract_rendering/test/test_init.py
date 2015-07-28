import abstract_rendering as ar
import unittest
import os

class InitTests(unittest.TestCase):
    def test_version_report(self):
        version_file = os.path.join(os.path.dirname(__file__), '', '../_version.txt')
        with open(version_file) as f:
            expected = f.readline().strip()
        self.assertEquals(ar.__version__, expected)

        parts = dict(zip(["major", "minor", "micro"],
                         map(int, expected.split("."))))
        self.assertEquals(ar.__version_info__, parts)
