#
# This code is a modified version of CEDR:
# https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import torch

import flexneuart.models.base_bert_split_max_chunk
from flexneuart.models import register
import flexneuart.models.base_bert as base_bert

VANILLA_BERT='vanilla_bert'


@register(VANILLA_BERT)
class VanillaBertRanker(flexneuart.models.base_bert_split_max_chunk.BertSplitMaxChunkRanker):
    """
        Vanilla BERT Ranker.

        Nogueira, Rodrigo, and Kyunghyun Cho. "Passage Re-ranking with BERT."
        arXiv preprint arXiv:1901.04085 (2019).

    """

    def __init__(self, bert_flavor, dropout=base_bert.DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        out = self.cls(self.dropout(cls_reps[-1]))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)
