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

from flexneuart.models import register
from flexneuart.models.utils import is_longformer
from flexneuart.models.base_bert import DEFAULT_BERT_DROPOUT

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from flexneuart.models.base_bert import BertBaseRanker


@register('longformer')
class LongformerRanker(BertBaseRanker):
    """
        The ranking transformer code, which is specific to Longformer:

        Beltagy, I., Peters, M.E. and Cohan, A., 2020. Longformer:
        The long-document transformer. arXiv preprint arXiv:2004.05150.
    """

    def __init__(self, bert_flavor='allenai/longformer-base-4096', dropout=DEFAULT_BERT_DROPOUT):
        """Constructor.

            :param bert_flavor:   the name of a specific longformer variant. As a sanity check we
                                  ensure that it contains a substring 'longformer'

        """
        super().__init__(bert_flavor)
        assert is_longformer(bert_flavor), 'bert_flavor_parameter: you should use one of the Longformer models'
        self.dropout = torch.nn.Dropout(dropout)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def encode_lf(self, query_tok, query_mask, doc_tok, doc_mask):
        """
            This function applies BERT to a query concatentated with a document.
            If this concatenation is too long, the document is truncated.

        :param query_tok:       batched and encoded query tokens
        :param query_mask:      query token mask (0 for padding, 1 for actual tokens)
        :param doc_tok:         batched and encoded document tokens
        :param doc_mask:        document token mask (0 for padding, 1 for actual tokens)

        :return: CLOS token representation.

        """
        batch_qty, max_qlen = query_tok.shape
        _, max_dlen = doc_tok.shape

        max_dlen = min(max_dlen, self.MAXLEN - 3 - max_qlen)
        assert max_dlen > 0, "Queries are too long!"

        # If documents are too long, they need to be shortened
        doc_toks = doc_tok[:max_dlen]
        doc_masks = doc_mask[:max_dlen]

        CLSS = torch.full_like(query_tok[:, :1], self.CLS_TOK_ID)
        SEPS = torch.full_like(query_tok[:, :1], self.SEP_TOK_ID)
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build input sequences
        toks = torch.cat([CLSS, query_tok, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_masks, ONES], dim=1)

        segment_ids = torch.cat([NILS] * (2 + max_qlen) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)

        # execute BERT model
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = \
            self.bert(input_ids=toks,
                      token_type_ids=None,
                      attention_mask=mask.long(),
                      # The global attention mask is set to 1 only for query tokens,
                      # which is how it is done for the QA task (which should be similar to IR in this respect)
                      global_attention_mask = ((1-segment_ids)*mask).long(),
                      output_hidden_states=False)

        return outputs.last_hidden_state[ :, 0]

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps = self.encode_lf(query_tok, query_mask, doc_tok, doc_mask)
        out = self.cls(self.dropout(cls_reps))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)
