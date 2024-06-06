#!/usr/bin/env python
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
import torch

from typing import List

from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from flexneuart.config import BERT_BASE_MODEL
from flexneuart import models
from flexneuart.models.base_bert import BertBaseRanker
from flexneuart.models.base_bert import DEFAULT_BERT_DROPOUT
from flexneuart.models.utils import mean_pool


@models.register(models.VANILLA_BERT + '_stand')
class VanillaBertStandard(BertBaseRanker):
    """
        A standard vanilla BERT Ranker, which does not pad queries (unlike CEDR version of FirstP).

        Nogueira, Rodrigo, and Kyunghyun Cho. "Passage Re-ranking with BERT."
        arXiv preprint arXiv:1901.04085 (2019).

    """
    def __init__(self, bert_flavor=BERT_BASE_MODEL, dropout=DEFAULT_BERT_DROPOUT,
                 use_mean_pool=False, use_trust_remote_code: bool = False):
        super().__init__(bert_flavor, use_trust_remote_code=use_trust_remote_code)
        self.dropout = torch.nn.Dropout(dropout)
        self.use_mean_pool = use_mean_pool
        print('Dropout', self.dropout, 'use_mean_pool:', use_mean_pool)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def featurize(self, max_query_len : int, max_doc_len : int,
                        query_texts : List[str],
                        doc_texts : List[str]) -> tuple:
        """
           "Featurizes" input. This function itself create a batch
            b/c training code does not use a standard PyTorch loader!
        """
        tok : PreTrainedTokenizerBase = self.tokenizer
        assert len(query_texts) == len(doc_texts), \
            "Document array length must be the same as query array length!"
        input_list = list(zip(query_texts, doc_texts))


        # With only_second truncation, sometimes this will fail b/c the first sequence is already longer
        # than the maximum allowed length so batch_encode_plus will fail with an exception.
        # In many IR applications, a document is much longer than a string, so effectively
        # only_second truncation strategy will be used most of the time.
        res : BatchEncoding = tok.batch_encode_plus(batch_text_or_text_pairs=input_list,
                                   padding='longest',
                                   #truncation='only_second',
                                   truncation='longest_first',
                                   max_length=3 + max_query_len + max_doc_len,
                                   return_tensors='pt')

        # token_type_ids may be missing
        return (res.input_ids, getattr(res, "token_type_ids", None), res.attention_mask)

    def forward(self, input_ids, token_type_ids, attention_mask):
        kwargs = {}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = \
                self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if not self.use_mean_pool:
            cls_reps = outputs.last_hidden_state[:, 0]
        else:
            cls_reps = mean_pool(outputs.last_hidden_state, attention_mask)
        out = self.cls(self.dropout(cls_reps))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)
