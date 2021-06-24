#
# This code is based on CEDR: https://github.com/Georgetown-IR-Lab/cedr
# It has some modifications/extensions and it relies on our custom BERT
# library: https://github.com/searchivarius/pytorch-pretrained-BERT-mod
# (c) Georgetown IR lab & Carnegie Mellon University
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
#from pytools import memoize_method
import torch
import torch.nn.functional as F
import pytorch_pretrained_bert

import scripts.cedr.modeling_util as modeling_util
from scripts.config import BERT_LARGE_MODEL, BERT_BASE_MODEL

USE_BATCH_COEFF=True
DEFAULT_BERT_DROPOUT=0.1


def init_bert_params(obj_ref, is_large):
    if not is_large:
        obj_ref.BERT_MODEL = BERT_BASE_MODEL
        obj_ref.CHANNELS = 12 + 1  # from bert-base-uncased
        obj_ref.BERT_SIZE = 768  # from bert-base-uncased
    else:
        obj_ref.BERT_MODEL = BERT_LARGE_MODEL
        obj_ref.CHANNELS = 24 + 1  # from bert-base-uncased
        obj_ref.BERT_SIZE = 1024  # from bert-base-uncased

    print('Model type:', obj_ref.BERT_MODEL,
          '# of channels:', obj_ref.CHANNELS,
          'hidden layer size:', obj_ref.BERT_SIZE)


class BertRanker(torch.nn.Module):
    def __init__(self, bert_large):
        """Bert ranker constructor

        :param bert_large: True if we need the large BERT model.
                         otherwise, the base version would be used.
        """
        super().__init__()
        init_bert_params(self, bert_large)

        # Large and base BERT have the same tokenizers:
        # https://github.com/huggingface/transformers/issues/424

        self.bert = modeling_util.CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(self.BERT_MODEL)

    def set_grad_checkpoint_param(self, param):
        self.bert.set_grad_checkpoint_param(param)

    def forward(self, **inputs):
        raise NotImplementedError

    #@memoize_method
    # memoization of the tokenizer is not very useful with large datasets,
    # but it can cause a huge memory bloating
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        batch_qty, max_qlen = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        max_doc_tok_len = maxlen - max_qlen - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, max_doc_tok_len)
        doc_masks, _ = modeling_util.subbatch(doc_mask, max_doc_tok_len)
        if USE_BATCH_COEFF:
          batch_coeff = modeling_util.get_batch_avg_coeff(doc_mask, max_doc_tok_len)
          batch_coeff = batch_coeff.view(batch_qty, 1)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_masks, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + max_qlen) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)

        # execute BERT model
        result = self.bert(toks, segment_ids.long(), mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:batch_qty, 1:max_qlen+1] for r in result]
        doc_results = [r[:, max_qlen+2:-1] for r in result]
        # TODO it's better to pass sbcount to this function rather than make
        #      un_subbatch recomputed sbcount again from max_doc_tok_len
        doc_results = [modeling_util.un_subbatch(r, doc_tok, max_doc_tok_len) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // batch_qty):
                cls_result.append(cls_output[i*batch_qty:(i+1)*batch_qty])
            # Leonid Boytsov: though proper batch-specific scaling should be 
            # be necessary, there's in practice no deterioration due to
            # imperfect scaling introduced by simple averaging. Furthermore,
            # on 11 GB gpus, one cannot have a mini-batch of the size > 1
            # (larger batches are simulated via graident accumulation)
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

    def __init__(self, bert_large=False, dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_large)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        out = self.cls(self.dropout(cls_reps[-1]))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)


class CedrPacrrRanker(BertRanker):
    def __init__(self, max_query_len, bert_large=False):
        super().__init__(bert_large)
        QLEN = max_query_len
        KMAX = 2
        NFILTERS = 32
        MINGRAM = 1
        MAXGRAM = 3
        self.simmat = modeling_util.SimmatModule()
        self.ngrams = torch.nn.ModuleList()
        self.rbf_bank = None
        for ng in range(MINGRAM, MAXGRAM+1):
            ng = modeling_util.PACRRConvMax2dModule(ng, NFILTERS, k=KMAX, channels=self.CHANNELS)
            self.ngrams.append(ng)
        qvalue_size = len(self.ngrams) * KMAX
        self.linear1 = torch.nn.Linear(self.BERT_SIZE + QLEN * qvalue_size, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        scores = [ng(simmat) for ng in self.ngrams]
        scores = torch.cat(scores, dim=2)
        scores = scores.reshape(scores.shape[0], scores.shape[1] * scores.shape[2])
        scores = torch.cat([scores, cls_reps[-1]], dim=1)
        rel = F.relu(self.linear1(scores))
        rel = F.relu(self.linear2(rel))
        rel = self.linear3(rel)
        # the last dimension is singleton and needs to be removed
        return rel.squeeze(dim=-1)


class CedrKnrmRanker(BertRanker):
    def __init__(self, bert_large=False):
        super().__init__(bert_large)
        MUS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.bert_ranker = VanillaBertRanker()
        self.simmat = modeling_util.SimmatModule()
        self.kernels = modeling_util.KNRMRbfKernelBank(MUS, SIGMAS)
        self.combine = torch.nn.Linear(self.kernels.count() * self.CHANNELS + self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                       .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                       .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        mask = (simmat.sum(dim=3) != 0.) # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2) # sum over query terms
        result = torch.cat([result, cls_reps[-1]], dim=1)
        scores = self.combine(result) # linear combination over kernels
        # the last dimension is singleton and needs to be removed
        return scores.squeeze(dim=-1)


class CedrDrmmRanker(BertRanker):
    def __init__(self, bert_large=False):
        super().__init__(bert_large)
        NBINS = 11
        HIDDEN = 5
        self.bert_ranker = VanillaBertRanker()
        self.simmat = modeling_util.SimmatModule()
        self.histogram = modeling_util.DRMMLogCountHistogram(NBINS)
        self.hidden_1 = torch.nn.Linear(NBINS * self.CHANNELS + self.BERT_SIZE, HIDDEN)
        self.hidden_2 = torch.nn.Linear(HIDDEN, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        histogram = self.histogram(simmat, doc_tok, query_tok)
        BATCH, CHANNELS, QLEN, BINS = histogram.shape
        histogram = histogram.permute(0, 2, 3, 1)
        output = histogram.reshape(BATCH * QLEN, BINS * CHANNELS)
        # repeat cls representation for each query token
        cls_rep = cls_reps[-1].reshape(BATCH, 1, -1).expand(BATCH, QLEN, -1).reshape(BATCH * QLEN, -1)
        output = torch.cat([output, cls_rep], dim=1)
        term_scores = self.hidden_2(torch.relu(self.hidden_1(output))).reshape(BATCH, QLEN)
        return term_scores.sum(dim=1)

