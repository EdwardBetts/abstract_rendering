from __future__ import print_function
import unittest

def run_all(verbose=False):
    suite = unittest.defaultTestLoader.discover('.', pattern='*[T|t]est*.py')
    if verbose:
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        result = suite.run(unittest.TestResult())
        print(result)
    return 0 if result.wasSuccessful() else 1
