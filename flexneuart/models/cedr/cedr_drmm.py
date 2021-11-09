#
# This code is based on CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
from flexneuart.config import BERT_BASE_MODEL
from flexneuart.models import register

import torch

from flexneuart.models.base_bert_split_max_chunk import BertSplitMaxChunkRanker
from .utils import SimmatModule, DRMMLogCountHistogram


@register('cedr_drmm')
class CedrDrmmRanker(BertSplitMaxChunkRanker):
    """
        CEDR DRMM model.

        MacAvaney, Sean, et al. "CEDR: Contextualized embeddings for document ranking."
        Proceedings of the 42nd International ACM SIGIR Conference. 2019.

    """
    def __init__(self, bert_flavor=BERT_BASE_MODEL):
        super().__init__(bert_flavor)
        NBINS = 11
        HIDDEN = 5

        self.simmat = SimmatModule()
        self.histogram = DRMMLogCountHistogram(NBINS)
        self.hidden_1 = torch.nn.Linear(NBINS * self.CHANNELS + self.BERT_SIZE, HIDDEN)
        self.hidden_2 = torch.nn.Linear(HIDDEN, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        histogram = self.histogram(simmat, doc_tok, query_tok)
        BATCH, CHANNELS, QLEN, BINS = histogram.shape
        histogram = histogram.permute(0, 2, 3, 1)
        output = histogram.reshape(BATCH * QLEN, BINS * CHANNELS)
        # repeat cls representation for each query token
        cls_rep = cls_reps[-1].reshape(BATCH, 1, -1).expand(BATCH, QLEN, -1).reshape(BATCH * QLEN, -1)
        output = torch.cat([output, cls_rep], dim=1)
        term_scores = self.hidden_2(torch.relu(self.hidden_1(output))).reshape(BATCH, QLEN)
        return term_scores.sum(dim=1)

