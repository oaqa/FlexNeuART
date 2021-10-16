#!/usr/bin/env python
import unittest
from flexneuart.models.train.sampler import TrainSamplerFixedChunkSize, TrainSample

TRAIN_PAIRS={
    '0' : {'0' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '-1': -1},
    '1' : {'5' : 5, '6' : 6, '7' : 7, '8' : 8,'9' : 9},
    '2' : {'10' : 10, '11' : 11, '12' : 12, '13' : 13, '14' : 14}
}
QRELS={
   '0' : {'0' : 1},
   '1' : {'9' : 1},
   '2' : {'12' : 1}
}

class TestTrainSampler(unittest.TestCase):
    def test1(self):
        sampler = TrainSamplerFixedChunkSize(train_pairs=TRAIN_PAIRS, neg_qty_per_query=6, qrels=QRELS,
                                             do_shuffle=False)

        self.assertEqual(len(list(sampler)), 3)

    def test2(self):
        for shuffle in [True, False]:
            for neg_qty in range(1, 100):
                sampler = TrainSamplerFixedChunkSize(train_pairs=TRAIN_PAIRS, neg_qty_per_query=neg_qty, qrels=QRELS,
                                                     do_shuffle=shuffle)
                for qnum, obj in enumerate(sampler):
                    obj : TrainSample = obj
                    qid  = obj.qid
                    if not shuffle:
                        self.assertEqual(int(qid), qnum)
                    self.assertTrue(obj.pos_id in QRELS[qid] and QRELS[qid][obj.pos_id] == 1)
                    self.assertEqual(obj.pos_id_score, TRAIN_PAIRS[qid][obj.pos_id])

                    self.assertEqual(len(obj.neg_id_scores), neg_qty)
                    self.assertEqual(len(obj.neg_ids), neg_qty)
                    for k in range(neg_qty):
                        neg_id = obj.neg_ids[k]
                        neg_score = obj.neg_id_scores[k]
                        self.assertEqual(str(neg_score), neg_id)


if __name__ == "__main__":
    unittest.main()