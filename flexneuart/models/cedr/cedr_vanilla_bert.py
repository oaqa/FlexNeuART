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

from flexneuart.config import BERT_BASE_MODEL
from flexneuart import models
from flexneuart.models.base_bert_split_max_chunk import BertSplitMaxChunkRanker
from flexneuart.models.base_bert import DEFAULT_BERT_DROPOUT


@models.register(models.VANILLA_BERT)
class VanillaBertCEDRRanker(BertSplitMaxChunkRanker):
    """
        A CEDR variant of the vanilla BERT Ranker.

        It has several pecularities (but it is better than VanillaBertNoPadRanker):

        1. queries are padded to a given length.
        2. long documents that cannot be processed in "one go"
           are split into chunks and each chunk is encoded separately (with query prepended).
        3. the result token is the average CLS tokens from all these chunks.
        4. for short documents (and queries), this works like BERT FirstP, which pads queries.

        Nogueira, Rodrigo, and Kyunghyun Cho. "Passage Re-ranking with BERT."
        arXiv preprint arXiv:1901.04085 (2019).

        MacAvaney, Sean, et al. "CEDR: Contextualized embeddings for document ranking."
        Proceedings of the 42nd International ACM SIGIR Conference. 2019.

        Dai, Zhuyun, and Jamie Callan.
        "Deeper text understanding for IR with contextual neural language modeling."
        Proceedings of the 42nd International ACM SIGIR Conference. 2019.
    """
    def __init__(self, bert_flavor=BERT_BASE_MODEL, dropout=DEFAULT_BERT_DROPOUT):
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
