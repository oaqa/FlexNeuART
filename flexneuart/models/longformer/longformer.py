import torch
import torch.nn.functional as F

from flexneuart.config import BERT_BASE_MODEL
from flexneuart.models import register
from flexneuart.models.base_bert_split_slide_window import \
        BertSplitSlideWindowRanker, DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE
from flexneuart.models.base_bert import DEFAULT_BERT_DROPOUT

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from flexneuart.models import utils as modeling_util
from flexneuart.models.base_bert import BertBaseRanker

@register('longformer')
class LongformerRanker(BertBaseRanker):
    """
        This code I am going to try to base it off of the BERT one, but I am not sure if I can.
    """

    def __init__(self, bert_flavor='allenai/longformer-base-4096', dropout=DEFAULT_BERT_DROPOUT, att_window_size=512):
        """Constructor.

            :param bert_flavor:   the name of the underlying Transformer/BERT. Various
                                  Transformer models are possible as long as they return
                                  the object BaseModelOutputWithPoolingAndCrossAttentions.

        """
        super().__init__(bert_flavor)
        self.att_window_size = att_window_size # This window size should be even.
        self.dropout = torch.nn.Dropout(dropout)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def encode_lf(self, query_tok, query_mask, doc_tok, doc_mask):
        """
            This function applies BERT to a query concatentated with a document.
            If this concatenation is too long to fit into a specified window, the
            document is split into (possibly overlapping) chunks,
            and each chunks is encoded *SEPARATELY*.

            Afterwards, individual CLS representations can be combined in several ways
            as defined by self.cls_aggreg_type

        :param query_tok:       batched and encoded query tokens
        :param query_mask:      query token mask (0 for padding, 1 for actual tokens)
        :param doc_tok:         batched and encoded document tokens
        :param doc_mask:        document token mask (0 for padding, 1 for actual tokens)

        :return: combined CLS representations of the chunks.

        It is also worth noting that I am overriding this in order to avoid having to use the split slide window class.

        """
        batch_qty, max_qlen = query_tok.shape

        sb_count = 1
        sbcount = sb_count

        doc_toks = doc_tok

        doc_masks = doc_mask

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.CLS_TOK_ID)
        SEPS = torch.full_like(query_toks[:, :1], self.SEP_TOK_ID)
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_masks, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + max_qlen) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)

        # execute BERT model
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = \
            self.bert(input_ids=toks,
                      token_type_ids=None,
                      attention_mask=mask.long(),
                      global_attention_mask = ((1-segment_ids)*mask).long(),
                      output_hidden_states=True)
        result = outputs.hidden_states

        # build CLS representation
        cls_results = []
        last_layer = (result[-1])
        for layer in result:
            # Only the last layer is being processed, and we only need the CLS part of it from the 768 elements.
            cls_output = layer[:, 0]
            cls_results.append(cls_output)
        return cls_results

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps = self.encode_lf(query_tok, query_mask, doc_tok, doc_mask)
        out = self.cls(self.dropout(cls_reps[-1]))
        # the last dimension is singleton and needs to be removed
        print()
        return out.squeeze(dim=-1)
