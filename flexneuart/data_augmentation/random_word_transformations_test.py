from utils import *
import unittest
from transforms import *
from random_word_transformations import RandomDeletion

class TestRandomWordTransformations(unittest.TestCase):
    # def test_random_insertions(self):
    #     answer = True
    #     self.assertEqual(answer, True)
    def test_random_deletions(self):
        random_deletion = RandomDeletion(p = 0.1)
        print(random_deletion.augment())
        pass
    # def test_random_swap(self):
    #     pass

if __name__ == "__main__":
    unittest.main()
