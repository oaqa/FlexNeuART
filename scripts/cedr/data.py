#
# This code is a modified version of CEDR: https://github.com/Georgetown-IR-Lab/cedr
# (c) Georgetown IR lab & Carnegie Mellon University
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import random
from tqdm import tqdm
import torch
import pickle

from collections import Counter

DEFAULT_MAX_QUERY_LEN=32
DEFAULT_MAX_DOC_LEN=512 - DEFAULT_MAX_QUERY_LEN - 4


def read_datafiles(files):
    queries = {}
    docs = {}
    for file in files:
        for line in tqdm(file, desc='loading datafile (by line)', leave=False):
            cols = line.rstrip().split('\t')
            if len(cols) != 3:
                tqdm.write(f'skipping line: `{line.rstrip()}`')
                continue
            c_type, c_id, c_text = cols
            assert c_type in ('query', 'doc')
            if c_type == 'query':
                queries[c_id] = c_text
            if c_type == 'doc':
                docs[c_id] = c_text
    return queries, docs


def read_pairs_dict(file):
    result = {}
    for line in tqdm(file, desc='loading pairs (by line)', leave=False):
        qid, docid = line.split()
        result.setdefault(qid, {})[docid] = 1
    return result


def iter_train_pairs(model, device_name, dataset, train_pairs, do_shuffle, qrels,
                     batch_size, max_query_len, max_doc_len):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    for qid, did, query_tok, doc_tok in _iter_train_pairs(model, dataset, train_pairs, do_shuffle, qrels):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        if len(batch['query_id']) // 2 == batch_size:
            yield _pack_n_ship(batch, device_name, max_query_len, max_doc_len)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}


def train_item_qty(train_pairs):
    return len(list(train_pairs.keys()))


def _iter_train_pairs(model, dataset, train_pairs, do_shuffle, qrels):
    ds_queries, ds_docs = dataset
    while True:
        qids = list(train_pairs.keys())
        if do_shuffle:
            random.shuffle(qids)
        for qid in qids:
            pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) > 0]
            if len(pos_ids) == 0:
                continue
            pos_id = random.choice(pos_ids)
            pos_ids_lookup = set(pos_ids)

            neg_ids = [did for did in train_pairs[qid] if did not in pos_ids_lookup]
            if len(neg_ids) == 0:
                continue
            neg_id = random.choice(neg_ids)
            query_tok = model.tokenize(ds_queries[qid])
            pos_doc = ds_docs.get(pos_id)
            neg_doc = ds_docs.get(neg_id)
            if pos_doc is None:
                tqdm.write(f'missing doc {pos_id}! Skipping')
                continue
            if neg_doc is None:
                tqdm.write(f'missing doc {neg_id}! Skipping')
                continue
            yield qid, pos_id, query_tok, model.tokenize(pos_doc)
            yield qid, neg_id, query_tok, model.tokenize(neg_doc)


def iter_valid_records(model, device_name, dataset, run,
                       batch_size, max_query_len, max_doc_len):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    for qid, did, query_tok, doc_tok in _iter_valid_records(model, dataset, run):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        if len(batch['query_id']) == batch_size:
            yield _pack_n_ship(batch, device_name, max_query_len, max_doc_len)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    # final batch
    if len(batch['query_id']) > 0:
        yield _pack_n_ship(batch, device_name, max_query_len, max_doc_len)


def _iter_valid_records(model, dataset, run):
    ds_queries, ds_docs = dataset
    for qid in run:
        query_tok = model.tokenize(ds_queries[qid])
        for did in run[qid]:
            doc = ds_docs.get(did)
            if doc is None:
                tqdm.write(f'missing doc {did}! Skipping')
                continue
            doc_tok = model.tokenize(doc)
            yield qid, did, query_tok, doc_tok


def _pack_n_ship(batch, device_name, max_query_len, max_doc_len):
    dlen = min(max_doc_len, max(len(b) for b in batch['doc_tok']))
    return {
        'query_id': batch['query_id'],
        'doc_id': batch['doc_id'],
        'query_tok': _pad_crop(device_name, batch['query_tok'], max_query_len),
        'doc_tok': _pad_crop(device_name, batch['doc_tok'], dlen),
        'query_mask': _mask(device_name, batch['query_tok'], max_query_len),
        'doc_mask': _mask(device_name, batch['doc_tok'], dlen),
    }


def _pad_crop(device_name, items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [-1] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    res = torch.tensor(result).long()

    return res.to(device_name)


def _mask(device_name, items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = [1. for _ in item] + ([0.] * (l - len(item)))
        if len(item) >= l:
            item = [1. for _ in item[:l]]
        result.append(item)
    res = torch.tensor(result).float()

    return res.to(device_name)


# Create vocabulary from whilte-spaced text
class VocabBuilder:
    def __init__(self):
        self.total_counter = Counter()
        self.doc_counter = Counter()
        self.doc_qty = 0
        self.tot_qty = 0

    def procDoc(self, text):
        """White-space tokenize the document, update counters."""
        toks = text.strip().split()
        self.total_counter.update(toks)
        self.doc_counter.update(list(set(toks)))
        self.tot_qty += len(toks)
        self.doc_qty += 1

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            dt = [self.total_counter, self.doc_counter, self.doc_qty, self.tot_qty]
            pickle.dump(dt, f)

    @staticmethod
    def load(file_name):
        dt = []
        with open(file_name, 'rb') as f:
            dt = pickle.load(f)
        res = VocabBuilder()
        res.total_counter, res.doc_counter, res.doc_qty, res.tot_qty = dt
        return res


