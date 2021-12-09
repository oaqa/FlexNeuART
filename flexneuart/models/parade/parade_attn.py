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
import torch.nn.functional as F

from flexneuart.config import BERT_BASE_MODEL
from flexneuart.models import register
from flexneuart.models.base_bert_split_slide_window import \
        BertSplitSlideWindowRanker, DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE, \
        CLS_AGGREG_STACK
from flexneuart.models.base_bert import DEFAULT_BERT_DROPOUT


@register('parade_attn')
class ParadeAttentionRanker(BertSplitSlideWindowRanker):
    """
        PARADE attaention ranker. Contributed by Tianyi Lin

        Li, C., Yates, A., MacAvaney, S., He, B., & Sun, Y. (2020). PARADE:
        Passage representation aggregation for document reranking.
        arXiv preprint arXiv:2008.09093.
    """
    def __init__(self, bert_flavor=BERT_BASE_MODEL,
                 window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                 dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor, cls_aggreg_type=CLS_AGGREG_STACK,
                         window_size=window_size, stride=stride)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)
        self.att = torch.nn.Linear(self.BERT_SIZE, self.BERT_SIZE)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask).cls_results
        last_layer_cls_rep = torch.transpose(cls_reps[-1], 1, 2)
        att_weights = self.att(last_layer_cls_rep)
        att_weights = F.softmax(att_weights, dim=1)

        out = torch.sum(att_weights * last_layer_cls_rep, dim=1)
        out = self.cls(self.dropout(out))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)

