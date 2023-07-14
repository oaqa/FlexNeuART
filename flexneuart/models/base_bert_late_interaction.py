import torch

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


from flexneuart.models import utils as modeling_util
from flexneuart.models.base_bert import BertBaseRanker

DEFAULT_WINDOW_SIZE = 150
DEFAULT_STRIDE = 100


class BertLateInteraction(BertBaseRanker):
    def __init__(self, bert_flavor, window_size, stride):
        """Constructor.

            :param bert_flavor:     the name of the underlying Transformer/BERT. Various
                                    Transformer models are possible as long as they return
                                    the object BaseModelOutputWithPoolingAndCrossAttentions.
            :param window_size:     a size of the window (in # of tokens)
            :param stride:          a step


        """
        super().__init__(bert_flavor)
        self.window_size = window_size
        self.stride = stride

    def forward(self, **inputs):
        raise NotImplementedError

    def encode_bert(self, text_tok, text_mask, is_query):
        """
        This function applies BERT to a query or document text.
        If this concatenation is too long to fit into a specified window, the
        text is split into (possibly overlapping) chunks,
        and each chunk is encoded *SEPARATELY*.

        Afterwards, individual representations are returned for each token.

        :param text_tok:            batched and encoded tokens (queries or text)
        :param text_mask:           token mask (0 for padding, 1 for actual tokens)
        :param is_query:            True for queries and False otherwise

        :return: representations of each token in the last layer of encoded chunks.
        mask of the same size as the representation, and the number of subbatches
        """
        batch_qty, _ = text_tok.shape

        split_toks, subbatch_count_toks = modeling_util.sliding_window_subbatch(text_tok,
                                                                                self.window_size,
                                                                                self.stride)
        split_masks, subbatch_count_masks = modeling_util.sliding_window_subbatch(text_mask,
                                                                                  self.window_size,
                                                                                  self.stride)

        assert subbatch_count_toks == subbatch_count_masks

        CLSS = torch.full_like(split_toks[:, :1], self.CLS_TOK_ID)
        SEPS = torch.full_like(split_toks[:, :1], self.SEP_TOK_ID)
        ONES = torch.ones_like(split_masks[:, :1])
        NILS = torch.zeros_like(split_masks[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, SEPS, split_toks, SEPS], dim=1)
        mask = torch.cat([ONES, ONES, split_masks, ONES], dim=1)
        segment_ids = torch.cat([NILS] * 2 + [NILS if is_query else ONES] * (1 + split_toks.shape[1]), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        if self.no_token_type_ids:
            token_type_ids = None
        else:
            token_type_ids = segment_ids.long()

        # execute BERT model
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = \
            self.bert(input_ids=toks,
                      token_type_ids=token_type_ids,
                      attention_mask=mask,
                      output_hidden_states=True)

        last_state = outputs.last_hidden_state

        assert last_state.size(0) == subbatch_count_masks * batch_qty

        return last_state, mask, subbatch_count_toks