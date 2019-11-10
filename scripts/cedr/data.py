#
# This code is taken from CEDR: https://github.com/Georgetown-IR-Lab/cedr
# (c) Georgetown IR lab
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import random
from tqdm import tqdm
import torch


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


def read_qrels_dict(file):
    result = {}
    for line in tqdm(file, desc='loading qrels (by line)', leave=False):
        qid, _, docid, score = line.split()
        result.setdefault(qid, {})[docid] = int(score)
    return result


def read_run_dict(file):
    result = {}
    for line in tqdm(file, desc='loading run (by line)', leave=False):
        qid, _, docid, rank, score, _ = line.split()
        result.setdefault(qid, {})[docid] = float(score)
    return result


def read_pairs_dict(file):
    result = {}
    for line in tqdm(file, desc='loading pairs (by line)', leave=False):
        qid, docid = line.split()
        result.setdefault(qid, {})[docid] = 1
    return result


def iter_train_pairs(model, no_cuda, dataset, train_pairs, qrels, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    for qid, did, query_tok, doc_tok in _iter_train_pairs(model, dataset, train_pairs, qrels):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        if len(batch['query_id']) // 2 == batch_size:
            yield _pack_n_ship(batch, no_cuda)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}



def _iter_train_pairs(model, dataset, train_pairs, qrels):
    ds_queries, ds_docs = dataset
    while True:
        qids = list(train_pairs.keys())
        random.shuffle(qids)
        for qid in qids:
            pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) > 0]
            if len(pos_ids) == 0:
                continue
            pos_id = random.choice(pos_ids)
            pos_ids_lookup = set(pos_ids)
            pos_ids = set(pos_ids)
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


def iter_valid_records(model, no_cuda, dataset, run, batch_size):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    for qid, did, query_tok, doc_tok in _iter_valid_records(model, dataset, run):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        if len(batch['query_id']) == batch_size:
            yield _pack_n_ship(batch, no_cuda)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': []}
    # final batch
    if len(batch['query_id']) > 0:
        yield _pack_n_ship(batch, no_cuda)


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


def _pack_n_ship(batch, no_cuda):
    QLEN = 20
    MAX_DLEN = 800
    DLEN = min(MAX_DLEN, max(len(b) for b in batch['doc_tok']))
    return {
        'query_id': batch['query_id'],
        'doc_id': batch['doc_id'],
        'query_tok': _pad_crop(no_cuda, batch['query_tok'], QLEN),
        'doc_tok': _pad_crop(no_cuda, batch['doc_tok'], DLEN),
        'query_mask': _mask(no_cuda, batch['query_tok'], QLEN),
        'doc_mask': _mask(no_cuda, batch['doc_tok'], DLEN),
    }


def _pad_crop(no_cuda, items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [-1] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    res = torch.tensor(result).long()
    if not no_cuda:
        res = res.cuda()
    return res


def _mask(no_cuda, items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = [1. for _ in item] + ([0.] * (l - len(item)))
        if len(item) >= l:
            item = [1. for _ in item[:l]]
        result.append(item)
    res = torch.tensor(result).float()
    if not no_cuda:
        res = res.cuda()
    return res
