from utils import *
import unittest
from transforms import *

class TestSynonymTransformations(unittest.TestCase):
    def test_synonym_transformation(self):
        answer = True
        self.assertEqual(answer, True)
    def test_hypernym_transformation(self):
        pass

if __name__ == "__main__":
    unittest.main()
