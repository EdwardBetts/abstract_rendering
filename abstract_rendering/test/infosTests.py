from __future__ import print_function

import unittest
import abstract_rendering.infos as infos


class Const(unittest.TestCase):
    def test(self):
        info = infos.const(3)

        self.assertTrue(callable(info))
        self.assertEqual(3, info(3))
        self.assertEqual(3, info(23))
        self.assertEqual(3, info(None))
        self.assertEqual(3, info([]))
        self.assertEqual(3, info(object()))
        self.assertEqual(3, info(self))


class Val(unittest.TestCase):
    def test(self):
        info = infos.val(15)

        self.assertTrue(callable(info))
        self.assertEqual(3, info(3))
        self.assertEqual(0, info(0))
        self.assertEqual(15, info(None))


class ValAt(unittest.TestCase):
    def test(self):
        info = infos.valAt(3, "Nothing")

        self.assertTrue(callable(info))
        self.assertEqual(3, info([0, 1, 2, 3]))
        self.assertEqual("three", info(["zero", "one", "two", "three"]))
        self.assertEqual("Nothing", info(None))


class Key(unittest.TestCase):
    def test(self):
        info = infos.key("val", "seven")

        self.assertTrue(callable(info))
        self.assertEqual(3, info({"val": 3, "other": 6}))
        self.assertEqual("seven", info({"value": 3, "other": 6}))


class Attribute(unittest.TestCase):

    class Has():
        val = 13
        other = 6

    class HasNot():
        value = 13
        other = 6

    def test(self):
        info = infos.attribute("val", "seven")

        self.assertTrue(callable(info))
        self.assertEqual(13, info(self.Has()))
        self.assertEqual("seven", info(self.HasNot()))

class Encode(unittest.TestCase):
    def test(self):
        info = infos.encode(["zero", "one", "two"])

        self.assertEqual(0, info("zero"))
        self.assertEqual(1, info("one"))
        self.assertEqual(2, info("two"))
        self.assertEqual(3, info("stuff"))
        self.assertEqual(3, info("more"))

    def test_default(self):
        info = infos.encode(["other", "one", "two"], defcat=0)
        
        self.assertEqual(0, info("other"))
        self.assertEqual(1, info("one"))
        self.assertEqual(2, info("two"))
        self.assertEqual(0, info("stuff"))
        self.assertEqual(0, info("more"))


if __name__ == '__main__':
    unittest.main()
