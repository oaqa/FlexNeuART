import sys
sys.path.append('scripts')
from pytools import memoize_method
from collections import Counter
from data_convert.text_proc import *
from data_convert.convert_common import *

from pytools import memoize_method
import torch
import tqdm
import pickle

# Create vocabulary from text
class VocabBuilder:
  @staticmethod
  def tokenize(nlp, lemmatize, text):
    lemmas, unlemm = nlp.procText(text)
    return lemmas if lemmatize else unlemm

  def __init__(self, data_desc, dataset, lemmatize, stopwords):
    self.global_counter = Counter()
    self.local_counter = Counter()
    self.doc_qty = 0
    self.tot_qty = 0

    nlp = SpacyTextParser(SPACY_MODEL, stopwords, keepOnlyAlphaNum=True, lowerCase=True)

    with tqdm.tqdm(dataset, desc='Building vocabulary for: ' + data_desc) as pbar:
      for text in pbar:
        toks = list(VocabBuilder.tokenize(nlp, lemmatize, text))
        self.global_counter.update(toks)
        self.local_counter.update(list(set(toks)))
        self.tot_qty += len(toks)
        self.doc_qty += 1

  def save(self, file_name):
    with open(file_name, 'wb') as f:
      pickle.dump(self, f)

  @staticmethod
  def load(file_name):
    return pickle.load(open(file_name, 'rb'))



class Model1Ranker(torch.nn.Module):
  def __init__(self, dataset, lemmatize=False, stopwords=None, dim=128, top_k=50000, dropout=0.1):
    super().__init__()

    self.dataset = dataset
    self.doc_voc = None
    self.query_voc = None
    self.lemmatize = lemmatize
    self.dropout = dropout
    self.init_from_train(dataset, dim, top_k)
    self.stopwords = list(stopwords) if stopwords is not None else []
    self.nlp = SpacyTextParser(SPACY_MODEL, self.stopwords, keepOnlyAlphaNum=True, lowerCase=True)

  @memoize_method
  def tokenize(self, text):
    return VocabBuilder.tokenize(self.nlp, self.lemmatize, text)

  def _compile_vocab(self, data, pbar_desc):
    res = Counter()
    qty = 0
    with tqdm.tqdm(data, desc='Building vocabulary for: ' + pbar_desc) as pbar:
      for text in pbar:
        toks = list(set(self.tokenize(text)))
        res.update(toks)
        qty += 1

    return res, qty

  # 1. Gather collection statistics
  # 2. Initialize query & document embeddings
  #    bitext_data
  def init_from_train(self, bitext_data, dim, top_k):
    queries, docs = bitext_data
    # both queries and docs are dictionaries

  def load(self, path):
    raise NotImplemented
    #self.load_state_dict(torch.load(path), strict=False)
