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
import torch.nn.functional as F

from flexneuart.models.base_bert_split_max_chunk import BertSplitMaxChunkRanker
from .utils import SimmatModule, PACRRConvMax2dModule

@register('cedr_pacrr')
class CedrPacrrRanker(BertSplitMaxChunkRanker):
    """
        CEDR PACRR model.

        MacAvaney, Sean, et al. "CEDR: Contextualized embeddings for document ranking."
        Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2019.

    """
    def __init__(self, max_query_len, bert_flavor=BERT_BASE_MODEL):
        super().__init__(bert_flavor)
        QLEN = max_query_len
        KMAX = 2
        NFILTERS = 32
        MINGRAM = 1
        MAXGRAM = 3
        self.simmat = SimmatModule()
        self.ngrams = torch.nn.ModuleList()
        self.rbf_bank = None
        for ng in range(MINGRAM, MAXGRAM+1):
            ng = PACRRConvMax2dModule(ng, NFILTERS, k=KMAX, channels=self.CHANNELS)
            self.ngrams.append(ng)
        qvalue_size = len(self.ngrams) * KMAX
        self.linear1 = torch.nn.Linear(self.BERT_SIZE + QLEN * qvalue_size, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        scores = [ng(simmat) for ng in self.ngrams]
        scores = torch.cat(scores, dim=2)
        scores = scores.reshape(scores.shape[0], scores.shape[1] * scores.shape[2])
        scores = torch.cat([scores, cls_reps[-1]], dim=1)
        rel = F.relu(self.linear1(scores))
        rel = F.relu(self.linear2(rel))
        rel = self.linear3(rel)
        # the last dimension is singleton and needs to be removed
        return rel.squeeze(dim=-1)

