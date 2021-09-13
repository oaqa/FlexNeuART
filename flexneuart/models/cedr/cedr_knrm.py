#
# This code is based on CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
from flexneuart.models import register
from flexneuart.config import BERT_BASE_MODEL

import torch

from flexneuart.models.base_bert_split_max_chunk import BertSplitMaxChunkRanker
from .utils import SimmatModule, KNRMRbfKernelBank


@register('cedr_knrm')
class CedrKnrmRanker(BertSplitMaxChunkRanker):
    """
        CEDR KNRM model.

        MacAvaney, Sean, et al. "CEDR: Contextualized embeddings for document ranking."
        Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2019.

    """
    def __init__(self, bert_flavor=BERT_BASE_MODEL):
        super().__init__(bert_flavor)
        MUS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]

        self.simmat = SimmatModule()
        self.kernels = KNRMRbfKernelBank(MUS, SIGMAS)
        self.combine = torch.nn.Linear(self.kernels.count() * self.CHANNELS + self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                       .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                       .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        mask = (simmat.sum(dim=3) != 0.) # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2) # sum over query terms
        result = torch.cat([result, cls_reps[-1]], dim=1)
        scores = self.combine(result) # linear combination over kernels
        # the last dimension is singleton and needs to be removed
        return scores.squeeze(dim=-1)
