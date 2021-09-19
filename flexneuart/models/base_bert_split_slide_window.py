#
# This code is based on CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from flexneuart.models import utils as modeling_util
from flexneuart.models.base_bert import BertBaseRanker

CLS_AGGREG_STACK = 'cls_stack'
CLS_AGGREG_AVG = 'cls_avg'
CLS_AGGREG_MAX = 'cls_max'

DEFAULT_WINDOW_SIZE = 150
DEFAULT_STRIDE = 100


class BertSplitSlideWindowRanker(BertBaseRanker):
    """
        A BERT-based ranker that sub-batches long documents by splitting them into
        possibly overlapping chunks (i.e., we use a sliding window to construct them).
    """
    def __init__(self, bert_flavor, cls_aggreg_type, window_size, stride):
        """Constructor.

            :param bert_flavor:     the name of the underlying Transformer/BERT. Various
                                    Transformer models are possible as long as they return
                                    the object BaseModelOutputWithPoolingAndCrossAttentions.
            :param cls_aggreg_type  controls how chunk-specific CLS tokens are aggregated
                                    by the function forward
            :param window_size:     a size of the window (in # of tokens)
            :param stride:          a step


        """
        super().__init__(bert_flavor)
        self.window_size = window_size
        self.stride = stride
        assert cls_aggreg_type in [CLS_AGGREG_AVG, CLS_AGGREG_MAX, CLS_AGGREG_STACK], \
            "Incorrect CLS aggregation type: " + cls_aggreg_type

        self.cls_aggreg_type = cls_aggreg_type

    def forward(self, **inputs):
        raise NotImplementedError

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
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

        """
        batch_qty, max_qlen = query_tok.shape

        doc_toks, sbcount = modeling_util.sliding_window_subbatch(doc_tok,
                                                                  self.window_size, self.stride)
        doc_masks, _ = modeling_util.sliding_window_subbatch(doc_mask,
                                                             self.window_size, self.stride)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.CLS_TOK_ID)
        SEPS = torch.full_like(query_toks[:, :1], self.SEP_TOK_ID)
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_masks, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + max_qlen) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)

        # execute BERT model
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = \
            self.bert(input_ids=toks,
                      token_type_ids=segment_ids.long(),
                      attention_mask=mask,
                      output_hidden_states=True)
        result = outputs.hidden_states

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // batch_qty):
                cls_result.append(cls_output[i*batch_qty:(i+1)*batch_qty])

            if self.cls_aggreg_type == CLS_AGGREG_AVG:
                cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            elif self.cls_aggreg_type == CLS_AGGREG_MAX:
                cls_result, _ = torch.stack(cls_result, dim=2).max(dim=2)
            elif self.cls_aggreg_type == CLS_AGGREG_STACK:
                cls_result = torch.stack(cls_result, dim=2)
            else:
                raise Exception('Unsupported CLS aggregation type: ' + self.cls_aggreg_type)

            cls_results.append(cls_result)

        return cls_results
