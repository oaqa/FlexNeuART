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

    def is_listwise(self):
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

    def is_listwise(self):
        return True

    def __init__(self, margin):
        self.loss = torch.nn.MultiMarginLoss(margin, reduction='sum')

    def compute(self, scores):
        zeros = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        return self.loss.forward(scores, target=zeros)


class PairwiseMarginRankingLossWrapper:
    @staticmethod
    def name():
        return 'pairwise_margin'

    def is_listwise(self):
        return False

    def __init__(self, margin):
        self.loss = torch.nn.MarginRankingLoss(margin, reduction='sum')

    def compute(self, scores):
        pos_doc_scores = scores[:, 0]
        neg_doc_scores = scores[:, 1]
        ones = torch.ones_like(pos_doc_scores)
        return self.loss.forward(pos_doc_scores, neg_doc_scores, target=ones)


class PairwiseSoftmaxLoss:
    @staticmethod
    def name():
        return 'pairwise_softmax'

    def is_listwise(self):
        return False

    def compute(self, scores):
        return torch.sum(1. - scores.softmax(dim=1)[:, 0])  # pairwise softmax


LOSS_FUNC_LIST = [MultiMarginRankingLossWrapper.name(),
                  CrossEntropyLossWrapper.name(),
                  PairwiseMarginRankingLossWrapper.name(),
                  PairwiseSoftmaxLoss.name()]
