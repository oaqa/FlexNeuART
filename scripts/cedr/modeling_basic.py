#
# This code is a modified version of CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import torch


from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

import scripts.cedr.modeling_util as modeling_util

USE_BATCH_COEFF=True
DEFAULT_BERT_DROPOUT=0.1


def init_model(obj_ref, bert_flavor):
    """Instantiate a model, a tokenizer, and remember their parameters.

    :param obj_ref:       an object to initialize.
    :param bert_flavor:   the name of the underlying Transformer/BERT
    """

    obj_ref.BERT_MODEL = bert_flavor

    model = AutoModel.from_pretrained(bert_flavor)
    config = model.config
    obj_ref.bert = model
    obj_ref.config = config
    obj_ref.tokenizer = tokenizer = AutoTokenizer.from_pretrained(bert_flavor)

    obj_ref.CHANNELS = config.num_hidden_layers + 1
    obj_ref.BERT_SIZE = config.hidden_size
    obj_ref.MAXLEN = config.max_position_embeddings

    obj_ref.CLS_TOK_ID = tokenizer.cls_token_id
    obj_ref.SEP_TOK_ID = tokenizer.sep_token_id

    print('Model type:', obj_ref.BERT_MODEL,
          '# of channels:', obj_ref.CHANNELS,
          'hidden layer size:', obj_ref.BERT_SIZE,
          'input window size:', obj_ref.MAXLEN)


"""
   The base class for all Transformer-based ranking models. We generally consider these
   models as BERT-variants, hence, the name of the base class.
"""
class BertBaseRanker(torch.nn.Module):
    def __init__(self, bert_flavor):
        """Bert ranker constructor.

            :param bert_flavor:   the name of the underlying Transformer/BERT. Various
                                  Transformer models are possible as long as they return
                                  the object BaseModelOutputWithPoolingAndCrossAttentions.

        """
        super().__init__()
        init_model(self, bert_flavor)


    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def forward(self, **inputs):
        raise NotImplementedError


class BertRanker(BertBaseRanker):
    def __init__(self, bert_flavor):
        """Bert ranker constructor.

            :param bert_flavor:   the name of the underlying Transformer/BERT. Various
                                  Transformer models are possible as long as they return
                                  the object BaseModelOutputWithPoolingAndCrossAttentions.

        """
        super().__init__(bert_flavor)

    def forward(self, **inputs):
        raise NotImplementedError

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        """
            This function applies BERT to a query concatentated with a document.
            If this concatenation is too long to fit into a BERT window, the
            document is split into chunks, and each chunks is encoded *SEPARATELY*.
            Afterwards, individual representations are combined:
                1. CLS representations are averaged
                2. Query represenation is taken from the first chunk
                2. Document representations from different chunks are concatenated

        :param query_tok:       batched and encoded query tokens
        :param query_mask:      query token mask (0 for padding, 1 for actual tokens)
        :param doc_tok:         batched and encoded document tokens
        :param doc_mask:        document token mask (0 for padding, 1 for actual tokens)

        :return: a triple (averaged CLS representations for each layer,
                          encoded query tokens for each layer,
                          encoded document tokens for each layer)

        """
        batch_qty, max_qlen = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.MAXLEN
        max_doc_tok_len = maxlen - max_qlen - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, max_doc_tok_len)
        doc_masks,sbcount_ = modeling_util.subbatch(doc_mask, max_doc_tok_len)
        assert sbcount == sbcount_
        if USE_BATCH_COEFF:
          batch_coeff = modeling_util.get_batch_avg_coeff(doc_mask, max_doc_tok_len)
          batch_coeff = batch_coeff.view(batch_qty, 1)

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
        outputs : BaseModelOutputWithPoolingAndCrossAttentions = \
                            self.bert(input_ids=toks,
                                      segment_ids=segment_ids.long(),
                                      attention_mask=mask,
                                      output_hidden_states=True)
        result = outputs.hidden_states

        # extract relevant subsequences for query and doc
        query_results = [r[:batch_qty, 1:max_qlen+1] for r in result]
        doc_results = [r[:, max_qlen+2:-1] for r in result]

        doc_results = [modeling_util.un_subbatch(r, doc_tok, sbcount) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // batch_qty):
                cls_result.append(cls_output[i*batch_qty:(i+1)*batch_qty])
            #
            # Leonid Boytsov: The original CEDR code averages all CLS tokens
            # even though some documents in the batch may be much shorter than
            # others so these CLS tokens won't represent any real chunks.
            #
            # When USE_BATCH_COEFF is set to true (which is different from the
            # original CEDR code), such CLS tokens are ignored. In practice, there
            # seems to be little-to-no improvement compared to the original
            # CEDR code.
            #
            # Furthermore, on 11 GB gpus, which are still pretty common,
            # one cannot have a mini-batch of the size > 1
            # (larger batches are simulated via gradient accumulation),
            # so no actual sub-batch averaging is happening.
            #
            if USE_BATCH_COEFF:
                cls_result = torch.stack(cls_result, dim=2).sum(dim=2)
                assert(cls_result.size()[0] == batch_qty)
                assert(batch_coeff.size()[0] == batch_qty)

                cls_result *= batch_coeff
            else:
                cls_result = torch.stack(cls_result, dim=2).mean(dim=2)

            cls_results.append(cls_result)

        return cls_results, query_results, doc_results


class VanillaBertRanker(BertRanker):
    """
        Vanilla BERT Ranker.

        Nogueira, Rodrigo, and Kyunghyun Cho. "Passage Re-ranking with BERT."
        arXiv preprint arXiv:1901.04085 (2019).

    """
    def __init__(self, bert_flavor, dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        out = self.cls(self.dropout(cls_reps[-1]))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)
