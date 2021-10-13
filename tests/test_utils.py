#!/usr/bin/env python
import unittest

from flexneuart.text_proc import handle_case

DEBUG_FAIL=False

class TestHandleCase(unittest.TestCase):
    def test1(self):
        self.assertEqual(handle_case(True, None), '')
        self.assertEqual(handle_case(False, None), '')

    def test2(self):
        self.assertEqual(handle_case(True, 'This is A Test'), 'this is a test')
        self.assertEqual(handle_case(False,'This is A Test'), 'This is A Test')

    def test3(self):
        self.assertTrue(not DEBUG_FAIL)


if __name__ == "__main__":
    unittest.main()
