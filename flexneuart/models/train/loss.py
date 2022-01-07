#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

#
# Various ranking losses
#
import torch

"""
   *IMPORTANT NOTE*: all the losses should have a reduction type sum!

   This is a wrapper class for the cross-entropy loss. It expects
   positive/negative-document scores arranged in equal-sized tuples, where
   the first score is for the positive document.

"""


class CrossEntropyLossWrapper:
    @staticmethod
    def name():
        return 'cross_entropy'

    def has_mult_negatives(self):
        return True

    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def compute(self, scores):
        zeros = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        return self.loss.forward(scores, target=zeros)


class MultiMarginRankingLossWrapper:
    @staticmethod
    def name():
        return 'multi_margin'

    def has_mult_negatives(self):
        return True

    def __init__(self, margin):
        self.loss = torch.nn.MultiMarginLoss(margin=margin, reduction='sum')

    def compute(self, scores):
        zeros = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        return self.loss.forward(scores, target=zeros)


class PairwiseMarginRankingLossWrapper:
    @staticmethod
    def name():
        return 'pairwise_margin'

    def has_mult_negatives(self):
        return False

    def __init__(self, margin):
        self.loss = torch.nn.MarginRankingLoss(margin=margin, reduction='sum')

    def compute(self, scores):
        pos_doc_scores = scores[:, 0]
        neg_doc_scores = scores[:, 1]
        ones = torch.ones_like(pos_doc_scores)
        return self.loss.forward(pos_doc_scores, neg_doc_scores, target=ones)


class OldPairwiseSoftmaxLoss:
    """This is pairwise (positive-negative pairs) softmax version that
       supports only a *SINGLE* negative per query.
       This older version (non-listwise) is kept only for testing purposes.
    """
    @staticmethod
    def name():
        return 'old_pairwise_softmax'

    def has_mult_negatives(self):
        return False

    def compute(self, scores):
        return torch.sum(1. - scores.softmax(dim=1)[:, 0])  # pairwise softmax


class PairwiseSoftmaxLoss:
    """This is pairwise (positive-negative pairs) softmax version that
       supports multiple negatives per query.
    """
    @staticmethod
    def name():
        return 'pairwise_softmax'

    def has_mult_negatives(self):
        return True

    def compute(self, scores):
        assert len(scores.shape) == 2

        batch_size, qty = scores.shape

        x0 = scores.unsqueeze(dim=-1)
        assert x0.shape == (batch_size, qty, 1)
        x1 = scores.unsqueeze(dim=-2)
        assert x1.shape == (batch_size, 1, qty)

        x0 = x0.expand(batch_size, qty, qty)
        x1 = x1.expand(batch_size, qty, qty)

        assert len(x0.shape) == 3
        assert len(x1.shape) == 3

        x_merged = torch.stack([x0, x1], dim=3)

        assert x_merged.shape == (batch_size, qty, qty, 2)

        x_softmax = 1. - x_merged.softmax(dim=-1)[:, :, :, 0]

        assert len(x_softmax.shape) == 3

        x_softmax = x_softmax[:, 0, 1:]
        assert len(x_softmax.shape) == 2

        # We average over all negatives
        return torch.sum(x_softmax) / float(x_softmax.shape[-1])

class RankNetLoss:
    """Burges, Chris, et al. "Learning to rank using gradient descent."
       Proceedings of the 22nd international conference on Machine learning. 2005.
    """
    @staticmethod
    def name():
        return 'ranknet'

    def has_mult_negatives(self):
        return True

    def compute(self, scores):
        assert len(scores.shape) == 2

        batch_size, qty = scores.shape

        x0 = scores.unsqueeze(dim=-1)
        assert x0.shape == (batch_size, qty, 1)
        x1 = scores.unsqueeze(dim=-2)
        assert x1.shape == (batch_size, 1, qty)

        x0 = x0.expand(batch_size, qty, qty)
        x1 = x1.expand(batch_size, qty, qty)

        assert len(x0.shape) == 3
        assert len(x1.shape) == 3

        x_merged = torch.stack([x0, x1], dim=3)

        assert x_merged.shape == (batch_size, qty, qty, 2)

        loss = -torch.nn.functional.log_softmax(x_merged, dim=-1)[:, :, :, 0]

        assert len(loss.shape) == 3

        loss = loss[:, 0, 1:]
        assert len(loss.shape) == 2

        # We average over all negatives
        return torch.sum(loss) / float(loss.shape[-1])


LOSS_FUNC_LIST = [MultiMarginRankingLossWrapper.name(),
                  CrossEntropyLossWrapper.name(),
                  PairwiseMarginRankingLossWrapper.name(),
                  PairwiseSoftmaxLoss.name(),
                  RankNetLoss.name()]
