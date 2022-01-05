#!/usr/bin/env python
import unittest

from flexneuart.text_proc.parse import KrovetzStemParser


class TestKrovetzStemParser(unittest.TestCase):
    # This are a very basic tests, but they are functional
    def basic_tests(self):
        parser = KrovetzStemParser(['is', 'a'])

        self.assertEqual(parser('This IS a simplest tests'), 'this simplest test')
        self.assertEqual(parser('This IS a simplest teStEd'), 'this simplest test')
        self.assertEqual(parser('This IS a simplest-teStEd'), 'this simplest test')
        self.assertEqual(parser('This IS a simplest#teStEd'), 'this simplest test')


if __name__ == "__main__":
    unittest.main()
