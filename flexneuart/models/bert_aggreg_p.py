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

from flexneuart.config import BERT_BASE_MODEL
from flexneuart.models import register
from flexneuart.models.base_bert_split_slide_window import \
        BertSplitSlideWindowRanker, CLS_AGGREG_STACK, DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE
from flexneuart.models.base_bert import DEFAULT_BERT_DROPOUT

BERT_MAXP='bert_maxp'
BERT_SUMP='bert_sump'


class BertAggregPRanker(BertSplitSlideWindowRanker):
    """
        BERT MaxP & SumP rankers. Contributed by Fangwei Gao.

        Dai, Zhuyun, and Jamie Callan.
        "Deeper text understanding for IR with contextual neural language modeling." SIGIR. 2019.
    """
    def __init__(self, bert_flavor, aggreg_type,
                 window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                 dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor, cls_aggreg_type=CLS_AGGREG_STACK,
                         window_size=window_size, stride=stride)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        assert aggreg_type in [BERT_MAXP, BERT_SUMP], \
            f'Unsupported aggregation type: {aggreg_type}'

        self.aggreg_type = aggreg_type
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask).cls_results
        last_layer_cls_rep = torch.transpose(cls_reps[-1], 1, 2)
        out = self.cls(self.dropout(last_layer_cls_rep))

        if self.aggreg_type == BERT_MAXP:
            out, _ = out.squeeze(dim=-1).max(dim=1)
        elif self.aggreg_type == BERT_SUMP:
            out, _ = out.squeeze(dim=-1).sum(dim=1)
        else:
            raise Exception(f'Unsupported aggregation type: {self.aggreg_type}')

        return out


@register(BERT_MAXP)
class BertMaxPRanker(BertAggregPRanker):
    def __init__(self, bert_flavor=BERT_BASE_MODEL,
                 window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                 dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor=bert_flavor,
                         aggreg_type=BERT_MAXP,
                         window_size=window_size, stride=stride, dropout=dropout)

@register(BERT_SUMP)
class BertSumPRanker(BertAggregPRanker):
    def __init__(self, bert_flavor=BERT_BASE_MODEL,
                 window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                 dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor=bert_flavor,
                         aggreg_type=BERT_SUMP,
                         window_size=window_size, stride=stride, dropout=dropout)
