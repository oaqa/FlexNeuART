#
# This code is a modified version of CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
import random
from tqdm import tqdm
import torch
import pickle

from collections import Counter

QUERY_ID_FIELD = 'query_id'
DOC_ID_FIELD = 'doc_id'
CAND_SCORE_FIELD = 'cand_score'
QUERY_TOK_FIELD = 'query_tok'
DOC_TOK_FIELD = 'doc_tok'
QUERY_MASK_FIELD = 'query_mask'
DOC_MASK_FIELD = 'doc_mask'

PAD_CODE=1 # your typical padding symbol
DEFAULT_MAX_QUERY_LEN=32
DEFAULT_MAX_DOC_LEN=512 - DEFAULT_MAX_QUERY_LEN - 4


def read_datafiles(files):
    """Read train and test files.

    :param files:   an array of file objects, which represent queries or documents (in any order)
    :return: a dataset, which is tuple of two dictionaries representing queries and documents, respectively.
    """
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
    """Read training pairs and scores provided by a candidate generator.

    :param file: an open file, not a file name!
    :return:    Training pairs in the dictionary of dictionary formats.
                Candidate generator scores are
                values of the inner-most dictionary.
    """
    result = {}
    for ln, line in enumerate(tqdm(file, desc='loading pairs (by line)', leave=False)):
        fields = line.split()
        if not len(fields) in [2, 3]:
            raise Exception(f'Wrong # of fields {len(fields)} in file {file}, line #: {ln+1}')
        qid, docid = fields[0: 2]
        if len(fields) == 3:
            score = fields[2]
        else:
            score = 0

        result.setdefault(qid, {})[docid] = float(score)

    return result


def write_pairs_dict(train_pairs, file_name):
    """Write training pairs.

    :param train_pairs:   training data dictionary of dictionaries.
    :param file_name:     output file name
    """
    with open(file_name, 'w') as outf:
        for qid, docid_dict in train_pairs.items():
            for did, score in docid_dict.items():
                outf.write(f'{qid}\t{did}\t{score}\n')


def create_empty_batch():
    return {QUERY_ID_FIELD: [], DOC_ID_FIELD: [], CAND_SCORE_FIELD: [], QUERY_TOK_FIELD: [], DOC_TOK_FIELD: []}


def iter_train_data(model, device_name, dataset,
                      train_pairs, do_shuffle, neg_qty_per_query,
                      qrels,
                      batch_size, max_query_len, max_doc_len):
    """Training data iterator.

    :param model:           a model object
    :param device_name:     a device name
    :param dataset:         a dataset object: a tuple returned by read_datafiles
    :param train_pairs:     training pairs returned by read_pairs_dict
    :param do_shuffle:      True to shuffle
    :param neg_qty_per_query:   a number of negative examples in each query
    :param qrels:           a QREL dictionary returned by read_qrels_dict
    :param batch_size:      the size of the batch
    :param max_query_len:   max. query length
    :param max_doc_len:     max. document length

    :return:
    """
    batch = create_empty_batch()
    for qid, did, score, query_tok, doc_tok in _iter_train_data(model, dataset,
                                                                 train_pairs=train_pairs,
                                                                 do_shuffle=do_shuffle,
                                                                 neg_qty_per_query=neg_qty_per_query,
                                                                 qrels=qrels):
        batch[QUERY_ID_FIELD].append(qid)
        batch[DOC_ID_FIELD].append(did)
        batch[CAND_SCORE_FIELD].append(score)
        batch[QUERY_TOK_FIELD].append(query_tok)
        batch[DOC_TOK_FIELD].append(doc_tok)

        if len(batch[QUERY_ID_FIELD]) // (1 + neg_qty_per_query) == batch_size:
            yield _pack_n_ship(batch, device_name, max_query_len, max_doc_len)
            batch = create_empty_batch()


def train_item_qty_upper_bound(train_pairs):
    return len(list(train_pairs.keys()))


def _iter_train_data(model, dataset,
                    train_pairs, do_shuffle, neg_qty_per_query,
                    qrels):
    ds_queries, ds_docs = dataset
    while True:
        qids = list(train_pairs.keys())
        if do_shuffle:
            random.shuffle(qids)
        for qid in qids:
            query_train_pairs = train_pairs[qid]

            pos_ids = [did for did in query_train_pairs if qrels.get(qid, {}).get(did, 0) > 0]
            if len(pos_ids) == 0:
                continue
            pos_id = random.choice(pos_ids)
            pos_ids_lookup = set(pos_ids)

            neg_id_arr = [did for did in query_train_pairs if did not in pos_ids_lookup]
            if len(neg_id_arr) < neg_qty_per_query:
                continue

            pos_doc = ds_docs.get(pos_id)
            if pos_doc is None:
                tqdm.write(f'missing doc {pos_id}! Skipping')
                continue

            neg_data_arr = []
            sample_fail = False

            # sampling *WITHOUT* replacement
            for neg_id in random.sample(neg_id_arr, neg_qty_per_query):
                neg_doc = ds_docs.get(neg_id)

                if neg_doc is None:
                    tqdm.write(f'missing doc {neg_id}! Skipping')
                    sample_fail = True
                    break

                neg_data_arr.append( (neg_id, neg_doc) )

            if sample_fail:
                continue

            query_tok_ids = model.tokenize_and_encode(ds_queries[qid])

            yield qid, pos_id, query_train_pairs[pos_id], \
                  query_tok_ids, model.tokenize_and_encode(pos_doc)

            assert len(neg_data_arr) == neg_qty_per_query
            for neg_id, neg_doc in neg_data_arr:
                yield qid, neg_id, query_train_pairs[neg_id], \
                      query_tok_ids, model.tokenize_and_encode(neg_doc)


def iter_valid_records(model, device_name, dataset, run,
                       batch_size, max_query_len, max_doc_len):
    batch = create_empty_batch()
    for qid, did, score, query_tok, doc_tok in _iter_valid_records(model, dataset, run):
        batch[QUERY_ID_FIELD].append(qid)
        batch[DOC_ID_FIELD].append(did)
        batch[CAND_SCORE_FIELD].append(score)
        batch[QUERY_TOK_FIELD].append(query_tok)
        batch[DOC_TOK_FIELD].append(doc_tok)
        if len(batch[QUERY_ID_FIELD]) == batch_size:
            yield _pack_n_ship(batch, device_name, max_query_len, max_doc_len)
            batch = create_empty_batch()
    # final batch
    if len(batch[QUERY_ID_FIELD]) > 0:
        yield _pack_n_ship(batch, device_name, max_query_len, max_doc_len)


def _iter_valid_records(model, dataset, run):
    ds_queries, ds_docs = dataset
    for qid in run:
        query_tok_ids = model.tokenize_and_encode(ds_queries[qid])
        for did, score in run[qid].items():
            doc = ds_docs.get(did)
            if doc is None:
                tqdm.write(f'missing doc {did}! Skipping')
                continue
            doc_tok_ids = model.tokenize_and_encode(doc)
            yield qid, did, score, query_tok_ids, doc_tok_ids


def _pack_n_ship(batch, device_name, max_query_len, max_doc_len):
    dlen = min(max_doc_len, max(len(b) for b in batch[DOC_TOK_FIELD]))
    return {
        QUERY_ID_FIELD:     batch[QUERY_ID_FIELD],
        DOC_ID_FIELD:       batch[DOC_ID_FIELD],
        CAND_SCORE_FIELD:   torch.FloatTensor(batch[CAND_SCORE_FIELD]).to(device_name),
        QUERY_TOK_FIELD:    _pad_crop(device_name, batch[QUERY_TOK_FIELD], max_query_len),
        DOC_TOK_FIELD:      _pad_crop(device_name, batch[DOC_TOK_FIELD], dlen),
        QUERY_MASK_FIELD:   _mask(device_name, batch[QUERY_TOK_FIELD], max_query_len),
        DOC_MASK_FIELD:     _mask(device_name, batch[DOC_TOK_FIELD], dlen),
    }


def _pad_crop(device_name, items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [PAD_CODE] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    res = torch.tensor(result).long()

    return res.to(device_name)


def _mask(device_name, items, max_len):
    result = []
    for e in items:
        elen = min(len(e), max_len)
        result.append([1.] * elen + [0.]*(max_len - elen))

    res = torch.tensor(result).float()

    return res.to(device_name)


class VocabBuilder:
    """Compile a vocabulary together with token stat. from *WHITE-SPACE* tokenized text."""
    def __init__(self):
        self.total_counter = Counter()
        self.doc_counter = Counter()
        self.doc_qty = 0
        self.tot_qty = 0

    def proc_doc(self, text):
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
        with open(file_name, 'rb') as f:
            dt = pickle.load(f)
        res = VocabBuilder()
        res.total_counter, res.doc_counter, res.doc_qty, res.tot_qty = dt
        return res


