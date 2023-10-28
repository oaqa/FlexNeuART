#!/usr/bin/env python
import unittest

from flexneuart.text_proc.parse import KrovetzStemParser
from flexneuart.text_proc.parse import is_alpha_num

class TestIsAlphaNum(unittest.TestCase):
    def basic_tests(self):
      self.assertTrue(is_alpha_num('это'))
      self.assertTrue(is_alpha_num('это0123456789.'))
      self.assertTrue(is_alpha_num('это-тест'))
      self.assertTrue(is_alpha_num('это_тест'))
      self.assertTrue(is_alpha_num('这是一个测试'))

      self.assertTrue(not is_alpha_num('это '))
      self.assertTrue(not is_alpha_num(' это'))
      self.assertTrue(not is_alpha_num('это тест'))
      self.assertTrue(not is_alpha_num('это_тест 这是一个测试'))


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
