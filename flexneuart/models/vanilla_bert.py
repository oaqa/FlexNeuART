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

from typing import List

from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from flexneuart.config import BERT_BASE_MODEL
from flexneuart import models
from flexneuart.models.base_bert import BertBaseRanker
from flexneuart.models.base_bert import DEFAULT_BERT_DROPOUT


@models.register(models.VANILLA_BERT)
class VanillaBertRanker(BertBaseRanker):
    """
        A vanilla BERT Ranker, which does not padd queries.

        Nogueira, Rodrigo, and Kyunghyun Cho. "Passage Re-ranking with BERT."
        arXiv preprint arXiv:1901.04085 (2019).

    """
    def __init__(self, bert_flavor=BERT_BASE_MODEL, dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def featurize(self, max_query_len : int, max_doc_len : int,
                        query_texts : List[str],
                        doc_texts : List[str]) -> tuple:
        """
           "Featurizes" input. This function itself create a batch
            b/c training code does not use a standard PyTorch loader!
        """
        tok : PreTrainedTokenizerBase = self.tokenizer
        input_list = [ (q[0 : max_query_len], d[0 : max_doc_len]) for (q, d) in zip(query_texts, doc_texts)]

        res : BatchEncoding = tok.batch_encode_plus(batch_text_or_text_pairs=input_list,
                                   padding='longest',
                                   truncation=True,
                                   # Specifying max_length is a bit paranoid since we truncate
                                   # queries & docs in the above code
                                   max_length=3 + max_query_len + max_doc_len,
                                   return_tensors='pt')

        return (res.input_ids, res.token_type_ids, res.attention_mask)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = \
            self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_reps = outputs.last_hidden_state[:, 0]
        out = self.cls(self.dropout(cls_reps))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)
