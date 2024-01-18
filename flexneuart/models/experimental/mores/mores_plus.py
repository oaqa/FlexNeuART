from typing import Optional
import torch
import torch.nn as nn
import math

from flexneuart.config import BART_BASE_MODEL
from flexneuart.models import utils as modeling_util
from flexneuart.models.base_bart import BartBaseRanker

from .modeling_bart import BartDecoder

from flexneuart.models import register

from transformers.models.bart.modeling_bart import BartClassificationHead


DEFAULT_WINDOW_SIZE = 150
DEFAULT_STRIDE = 100
DEFAULT_BART_DROPOUT = 0.1
DEFAULT_OUTPUT_HIDDEN_STATE = True
DEFAULT_OUTPUT_ATTENTIONS = True
RAND_SPECIAL_INIT_DEFAULT=True
DEFAULT_USE_SEP=True


class BartMoresModule(BartBaseRanker):
    def __init__(self, bart_flavor, window_size,
                 stride, output_attentions,
                 output_hidden_states):
        """Constructor.
        :param bart_flavor:     the name of the underlying Transformer/BART.
        :param window_size:     a size of the window (in # of tokens)
        :param stride:          a step
        :param output_attentions:   whether to output attentions

        """
        super().__init__(bart_flavor)
        self.window_size = window_size
        self.stride = stride
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        decoder = BartDecoder(self.config, self.bart.shared)
        decoder.load_state_dict(self.bart.decoder.state_dict())
        self.bart.decoder = decoder

    def _prepare_tokens(self, text_tok, text_mask):
        """ This function applies BART to a query or document text.
        If this concatenation is too long to fit into a specified window, the
        text is split into (possibly overlapping) chunks.

        :param text_tok:            batched and encoded tokens (queries or text)
        :param text_mask:           token mask (0 for padding, 1 for actual tokens)

        :return: representations of each token in the last layer of encoded chunks,
        mask of the same size as the representation, and the number of subbatches.
        """

        split_toks, subbatch_count_toks = modeling_util.sliding_window_subbatch(text_tok,
                                                                                self.window_size,
                                                                                self.stride)
        split_masks, subbatch_count_masks = modeling_util.sliding_window_subbatch(text_mask,
                                                                                  self.window_size,
                                                                                  self.stride)

        assert subbatch_count_toks == subbatch_count_masks
        CLSS = torch.full_like(split_toks[:, :1], self.CLS_TOK_ID)
        EOSS = torch.full_like(split_toks[:, :1], self.EOS_TOK_ID)
        ONES = torch.ones_like(split_masks[:, :1])

        # build BART input sequences
        toks = torch.cat([CLSS, split_toks, EOSS], dim=1)
        mask = torch.cat([ONES, split_masks, ONES], dim=1)

        #toks =split_toks
        #mask = split_masks
        return toks, mask, subbatch_count_toks


@register('mores_plus')
class BartMoresRanker(BartMoresModule):
    def __init__(self, bart_flavor=BART_BASE_MODEL,
                 window_size=DEFAULT_WINDOW_SIZE,
                 stride=DEFAULT_STRIDE,
                 dropout=DEFAULT_BART_DROPOUT,
                 output_attentions=DEFAULT_OUTPUT_ATTENTIONS,
                 output_hidden_states=DEFAULT_OUTPUT_HIDDEN_STATE):

        """Constructor

        :param bart_flavor:     the name of the underlying Transformer/BART.
        :param window_size:     a size of the window (in # of tokens)
        :param stride:          a step
        :param dropout:         dropout probability
        :param output_attentions:   whether to output attentions
        :param output_hidden_states:    whether to output hidden states
        """

        super().__init__(bart_flavor=bart_flavor,
                 window_size=window_size,
                 stride=stride,
                 output_attentions=output_attentions,
                 output_hidden_states=output_hidden_states)

        self.classification_head = BartClassificationHead(input_dim=self.BART_SIZE,
                                                          inner_dim=self.BART_SIZE,
                                                          num_classes=1,
                                                          pooler_dropout=dropout,)
        # initialize classification head
        self.classification_head.apply(self.bart._init_weights)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        query_toks, query_masks, _ = self._prepare_tokens(query_tok, query_mask)
        doc_toks, doc_masks, _ = self._prepare_tokens(doc_tok, doc_mask)

        # Here we assume the inputs are already chunked
        encoder_outputs = self.bart.encoder(
            input_ids=doc_toks,
            attention_mask=doc_masks,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

        hiddens = encoder_outputs.last_hidden_state

        encoder_mask = doc_masks

        B, _ = query_toks.shape
        hiddens = hiddens.reshape((B, -1, hiddens.shape[-1]))
        encoder_mask = encoder_mask.reshape((B, -1))

        # Check for batch size consistency
        assert hiddens.size(0) == B
        assert encoder_mask.size(0) == B

        decoder_outputs = self.bart.decoder(
            input_ids=query_toks,
            attention_mask=query_masks,
            encoder_hidden_states=hiddens,
            encoder_attention_mask=encoder_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

        last_hidden_state = decoder_outputs.last_hidden_state

        eos_mask = torch.where(query_toks == self.config.eos_token_id, 1, 0).to(last_hidden_state.dtype)
        sentence_representation = (eos_mask.view(eos_mask.shape + (1,)) * last_hidden_state).sum(1)
        logits = self.classification_head(sentence_representation)

        return logits.squeeze(dim=-1)
