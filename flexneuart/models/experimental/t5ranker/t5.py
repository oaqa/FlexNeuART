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
from flexneuart.models.base_t5 import T5BaseRanker, DEFAULT_T5_DROPOUT

class T5RankerBase(T5BaseRanker):
    """
    Base class for longT5/T5-based encoder only or encoder-decoder based rankers .
    """

    def __init__(self, t5_flavor='google/long-t5-local-base'):
        """Constructor.

            :param bert_flavor:   the name of a specific longformer variant. As a sanity check we
                                  ensure that it contains a substring 'longformer'

        """
        super().__init__(t5_flavor)

    def prepare_toks(self, query_tok, query_mask, doc_tok, doc_mask):
        """
            This function applies T5 to a query concatentated with a document.
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

        EOSS = torch.full_like(query_tok[:, :1], self.EOS_TOK_ID)
        ONES = torch.ones_like(query_mask[:, :1])

        # build input sequences
        toks = torch.cat([query_tok, doc_toks, EOSS], dim=1)
        mask = torch.cat([query_mask, doc_masks, ONES], dim=1)
        return toks, mask


@register('t5_enc')
class T5EncRanker(T5RankerBase):
    """
    """

    def __init__(self, t5_flavor='google/long-t5-tglobal-large', dropout=DEFAULT_T5_DROPOUT):
        """Constructor.

            :param t5_flavor:   the name of a specific t5 variant.
            :param dropout:           dropout probability
        """
        super().__init__(t5_flavor)
        self.dropout = torch.nn.Dropout(dropout)
        self.cls = torch.nn.Linear(self.T5_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def encode_t5(self, query_tok, query_mask, doc_tok, doc_mask):
        """
            This function applies BERT to a query concatentated with a document.
            If this concatenation is too long, the document is truncated.

        :param query_tok:       batched and encoded query tokens
        :param query_mask:      query token mask (0 for padding, 1 for actual tokens)
        :param doc_tok:         batched and encoded document tokens
        :param doc_mask:        document token mask (0 for padding, 1 for actual tokens)

        :return: CLS token representation.

        """
        toks, mask = self.prepare_toks(query_tok, query_mask, doc_tok, doc_mask)
        outputs = self.t5.encoder(input_ids=toks)

        return outputs.last_hidden_state[ :, 0]

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps = self.encode_t5(query_tok, query_mask, doc_tok, doc_mask)
        out = self.cls(self.dropout(cls_reps))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)

@register('t5_encdec')
class T5EncDecRanker(T5RankerBase):
    """
    """

    def __init__(self, t5_flavor='google/long-t5-tglobal-large', dropout=DEFAULT_T5_DROPOUT):
        """Constructor.

            :param t5_flavor:   the name of a specific t5 variant.
            :param dropout:           dropout probability

        """
        super().__init__(t5_flavor)
        self.dropout = torch.nn.Dropout(dropout)
        custom_token = "<extra_id_10>"
        if "<extra_id_10>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(["<extra_id_10>"])
            self.t5.resize_token_embeddings(len(self.tokenizer))
        self.custom_token_id = self.tokenizer.get_vocab()[custom_token]
        self.cls = torch.nn.Linear(self.T5_SIZE, self.t5.config.vocab_size)
        torch.nn.init.xavier_uniform_(self.cls.weight)


    def enc_dec(self, query_tok, query_mask, doc_tok, doc_mask):
        """
            This function applies BERT to a query concatentated with a document.
            If this concatenation is too long, the document is truncated.

        :param query_tok:       batched and encoded query tokens
        :param query_mask:      query token mask (0 for padding, 1 for actual tokens)
        :param doc_tok:         batched and encoded document tokens
        :param doc_mask:        document token mask (0 for padding, 1 for actual tokens)

        :return: last hidden state.

        """
        toks, mask = self.prepare_toks(query_tok, query_mask, doc_tok, doc_mask)

        # execute BERT model
        encoder_outputs = self.t5.encoder(input_ids=toks,
                    attention_mask=mask.long())

        # create decoder input ids  full of <pad> tokens of shape (batch_size, 1)
        decoder_input_ids = torch.full((toks.shape[0], 1), self.tokenizer.pad_token_id, dtype=torch.long).to(toks.device)

        outputs = self.t5.decoder(input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=mask.long())

        return outputs.last_hidden_state

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        last_hidden_state = self.enc_dec(query_tok, query_mask, doc_tok, doc_mask)
        # Original paper does not have dropout here
        logits = self.cls(self.dropout(last_hidden_state))
        out = logits[:, :, self.custom_token_id]
        return out.squeeze(dim=-1)
