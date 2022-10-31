#!/usr/bin/env python
from torch import FloatTensor
import unittest
from flexneuart.models.train.loss import OldPairwiseSoftmaxLoss, PairwiseSoftmaxLoss

TEST_CASES = [
    [[5, 0.1]],
    [[5, 0.1], [3, 2]],
    [[5, 0.1], [3, 2], [1, 1]],
    [[2, 3.3], [3, 2], [1, 1]],
    [[2, 3.3], [3, 0], [1, 1]],
    [[2, 3.3], [3, 0], [1, 2]],
    [[2, 3.3], [3, 0], [2, 2]],
]

class TestLoss(unittest.TestCase):
    def test1(self):
        loss_old = OldPairwiseSoftmaxLoss()
        loss_new = PairwiseSoftmaxLoss()

        self.assertTrue(loss_new.has_mult_negatives())

        test_qty = len(TEST_CASES)

        for k in range(test_qty):
            input = FloatTensor(TEST_CASES[k])
            loss1 = loss_old.compute(input)
            loss2 = loss_new.compute(input)
            self.assertAlmostEqual(loss1, loss2, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
