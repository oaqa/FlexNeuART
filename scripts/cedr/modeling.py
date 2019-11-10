#
# This code is taken from CEDR: https://github.com/Georgetown-IR-Lab/cedr
# (c) Georgetown IR lab
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
from pytools import memoize_method
import torch
import torch.nn.functional as F
import pytorch_pretrained_bert
import modeling_util

class BertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1 # from bert-base-uncased
        self.BERT_SIZE = 768 # from bert-base-uncased
        self.bert = modeling_util.CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)

        # execute BERT model
        result = self.bert(toks, segment_ids.long(), mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN+1] for r in result]
        doc_results = [r[:, QLEN+2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results


class VanillaBertRanker(BertRanker):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        return self.cls(self.dropout(cls_reps[-1]))

class CedrPacrrRanker(BertRanker):
    def __init__(self):
        super().__init__()
        QLEN = 20
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
        return rel


class CedrKnrmRanker(BertRanker):
    def __init__(self):
        super().__init__()
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
        return scores


class CedrDrmmRanker(BertRanker):
    def __init__(self):
        super().__init__()
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

