#
# This code is based on CEDR: https://github.com/Georgetown-IR-Lab/cedr
# (c) Georgetown IR lab, which is distributed under the MIT License.
# MIT License is compatible with Apache 2 license for the code in this repo.
#
from pytools import memoize_method
import torch
import os
import torch.nn.functional as F
import pytorch_pretrained_bert
import modeling_util

#
# TODO: We will at some point need to refactor and extract hard-coded constants to a separate file
#
class BertEncoder(torch.nn.Module):
  '''
    This class uses a BERT model with an additional fully-connected layer applied
    to the output of the CLS token to produce a dense vector
    of a given dimensionality.
  '''
  def __init__(self, dim, dropout):
    super().__init__()

    self.BERT_MODEL = 'bert-base-uncased'
    self.CHANNELS = 12 + 1  # from bert-base-uncased
    self.BERT_SIZE = 768  # from bert-base-uncased
    self.bert = modeling_util.CustomBertModel.from_pretrained(self.BERT_MODEL)
    self.tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(self.BERT_MODEL)

    self.dropout = torch.nn.Dropout(dropout)
    # A fully-connected layer that transforms internal BERT representation
    self.fc = torch.nn.Linear(self.BERT_SIZE, dim)


  def forward(self, toks, mask):
      cls_reps, _ = self.encode_bert(toks, mask)
      return self.fc(self.dropout(cls_reps[-1]))

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

  def encode_bert(self, toks, mask):
    BATCH, LEN = toks.shape
    DIFF = 1  # = [CLS]
    maxlen = self.bert.config.max_position_embeddings
    MAX_TOK_LEN = maxlen - LEN - DIFF

    subbatch_toks, sbcount = modeling_util.subbatch(toks, MAX_TOK_LEN)
    subbatch_mask, _ = modeling_util.subbatch(mask, MAX_TOK_LEN)

    CLSS = torch.full_like(subbatch_toks[:, :1], self.tokenizer.vocab['[CLS]'])
    ONES = torch.ones_like(subbatch_mask[:, :1])

    # build BERT input sequences
    toks_4model = torch.cat([CLSS, subbatch_toks], dim=1)
    mask_4model = torch.cat([ONES, subbatch_mask], dim=1)
    # While encoding queries & documents separately we have only one type of the segment
    segment_ids = torch.zeros_like(mask_4model)
    # Original CEDR developer Sean MacAvaney's comment: remove padding (will be masked anyway)
    # Leo's comment: Although, we modified the token-merging procedure,
    # it is likely still a useful thing to do.
    toks_4model[toks_4model == -1] = 0

    assert(toks_4model.shape == mask_4model.shape)
    assert(segment_ids.shape == mask_4model.shape)

    # execute BERT model
    results = self.bert(toks_4model, segment_ids.long(), mask_4model)

    # extract relevant subsequences
    unsubbatch_results = [modeling_util.un_subbatch(r, toks, MAX_TOK_LEN) for r in results]

    # build aggregate CLS representation by averaging CLS representations within each subbatch
    cls_results = []
    for layer in unsubbatch_results:
      cls_output = layer[:, 0]
      cls_result = []
      for i in range(cls_output.shape[0] // BATCH):
        cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
      cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
      cls_results.append(cls_result)

    return cls_results, unsubbatch_results



class DuetBertRanker(torch.nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.query_encoder = BertEncoder(dim=dim, dropout=dropout)
        self.doc_encoder = BertEncoder(dim=dim, dropout=dropout)
        self.tokenizer = self.query_encoder.tokenizer

    # TODO this is a copy-paste of the BertRanker class tokenize
    #      can we somehow unify two classes?
    @memoize_method
    def tokenize(self, text):
      toks = self.tokenizer.tokenize(text)
      toks = [self.tokenizer.vocab[t] for t in toks]
      return toks

    def save(self, path):
        self.query_encoder.save(path + '_query')
        self.doc_encoder.save(path + '_doc')

    def load(self, path):
        query_path = path + '_query'
        doc_path = path + '_doc'
        if not os.path.exists(query_path) or not os.path.exists(doc_path):
          print('Trying to load LM-finetuned model')
          self.query_encoder.load(path)
          self.doc_encoder.load(path)
        else:
          self.query_encoder.load(query_path)
          self.doc_encoder.load(doc_path)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        query_vec = self.query_encoder(query_tok, query_mask)
        doc_vec = self.doc_encoder(doc_tok, doc_mask)
        # Returnining batched dot-product
        return torch.einsum('bi, bi -> b', query_vec, doc_vec)

