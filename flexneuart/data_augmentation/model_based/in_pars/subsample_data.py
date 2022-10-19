"""
This scripts mixes two training sets: The validation set is taken from the 1st one.
"""
import argparse
import sys
import os
import numpy as np

from flexneuart.io import open_with_default_enc

# This things are hard-coded and must match Java and shell scripts
DATA_QUERY = 'data_query.tsv'
DATA_DOCS = 'data_docs.tsv'
TRAIN_PAIRS = 'train_pairs.tsv'
TEST_RUN = 'test_run.txt'
QRELS = 'qrels.txt'

DATA_TYPE_QUERY = 'query'
DATA_TYPE_DOC = 'doc'

sys.path.append('.')

from flexneuart.io.train_data import read_pairs_dict, write_pairs_dict, read_datafiles
from flexneuart.io.runs import read_run_dict, write_run_dict
from flexneuart.io.qrels import read_qrels_dict, qrel_entry2_str, QrelEntry
from flexneuart.utils import set_all_seeds


def prefix_key(d, pref):
    return {pref + k : v for k, v in d.items()}


def prefix_key_val(d, pref):
    return {pref + k : prefix_key(v, pref) for k, v in d.items()}


def write_filtered_datafiles(out_f, data, data_type, id_filter_set):
    # File must be opened
    print(f'Writing to {out_f.name} type: {data_type}')
    qty = 0
    for id, v in data.items():
        if id in id_filter_set:
            out_f.write(f'{data_type}\t{id}\t{v}\n')
            qty += 1

    print(f'{qty} items written')


def write_filtered_train_pairs(out_fn, train_pairs_full, qid_filter_set):
    # File must be opened
    print(f'Writing train pairs to {out_fn}')
    qty = 0
    train_pairs_filtered = {}
    for qid, did_dict in train_pairs_full.items():
        if qid in qid_filter_set:
            train_pairs_filtered[qid] = did_dict
            qty += len(did_dict)

    write_pairs_dict(train_pairs_filtered, out_fn)

    print(f'# of queris in a full set: {len(train_pairs_full)} filtered set: {len(train_pairs_filtered)}')
    print(f'{qty} items written')


def write_filtered_qrels(out_f, qrels, qid_filter_set):
    print(f'Writing qrels to {out_f.name}')
    # File must be opened
    qty = 0
    for qid, did_rel_dict in qrels.items():
        if qid in qid_filter_set:
            for did, grade in did_rel_dict.items():
                e = QrelEntry(query_id=qid, doc_id=did, rel_grade=grade)
                out_f.write(qrel_entry2_str(e) + '\n')
                qty += 1

    print(f'{qty} items written')


def merge_dict(dict1, dict2):
    res_dict = {}

    for k1, v1 in dict1.items():
        res_dict[k1] = v1

    for k2, v2 in dict2.items():
        assert not k2 in res_dict, f'Repeating dictionary key: {k2}'
        res_dict[k2] = v2

    return res_dict


parser = argparse.ArgumentParser('Mix two training sets in CEDR format')

parser.add_argument('--dir1', type=str, required=True)
parser.add_argument('--out_pref1', type=str, default='c1_')

parser.add_argument('--dir2', type=str, required=True)
parser.add_argument('--out_pref2', type=str, default='c2_')

parser.add_argument('--dir_out', type=str, required=True)

parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--prob1', type=float, default=1)
parser.add_argument('--prob2', type=float, default=1)

args = parser.parse_args()
print(args)

set_all_seeds(args.seed)

assert args.prob1 > 0 and args.prob1 <= 1
assert args.prob2 > 0 and args.prob2 <= 1

os.makedirs(args.dir_out, exist_ok=True)

# Read convert & save the run
run1 = prefix_key_val(read_run_dict(os.path.join(args.dir1, TEST_RUN)), args.out_pref1)
write_run_dict(run1, os.path.join(args.dir_out, TEST_RUN))

queries1, data1 = read_datafiles([os.path.join(args.dir1, fn) for fn in [DATA_DOCS, DATA_QUERY]])

train_pairs1 = prefix_key_val(read_pairs_dict(os.path.join(args.dir1, TRAIN_PAIRS)), args.out_pref1)

queries1 = prefix_key(queries1, args.out_pref1)
qrels1 = prefix_key_val(read_qrels_dict(os.path.join(args.dir1, QRELS)), args.out_pref1)
data1 = prefix_key(data1, args.out_pref1)

queries2, data2 = read_datafiles([os.path.join(args.dir2, fn) for fn in [DATA_DOCS, DATA_QUERY]])

queries2 = prefix_key(queries2, args.out_pref2)
qrels2 = prefix_key_val(read_qrels_dict(os.path.join(args.dir2, QRELS)), args.out_pref2)
data2 = prefix_key(data2, args.out_pref2)

train_pairs2 = prefix_key_val(read_pairs_dict(os.path.join(args.dir2, TRAIN_PAIRS)), args.out_pref2)

doc_ids_orig = list(train_pairs1.values())
qid_orig = list(train_pairs1.keys())
doc_ids_inp = set(train_pairs2.values())

qid_to_write = []

num_pairs = len(doc_ids_orig)

for i in range(num_pairs):
    if doc_ids_orig[i] in doc_ids_inp:
        qid_to_write.append(qid_orig[i])

# qids1 = list(train_pairs1.keys())
# if args.prob1 < 1:
#     qids1 = list(np.random.choice(qids1, size=int(args.prob1 * len(qids1)), replace=False))

# print('Selected ', len(qids1), 'queries from ', len(train_pairs1))

# qids2 = list(train_pairs2.keys())
# if args.prob2 < 2:
#     qids2 = list(np.random.choice(qids2, size=int(args.prob2 * len(qids2)), replace=False))

# print('Selected ', len(qids2), 'queries from ', len(train_pairs2))

qids_all = set(qid_to_write)

dids_all1 = []
for qid, did_dict in train_pairs1.items():
    if qid in qids_all:
        dids_all1.extend(list(did_dict.keys()))
dids_all1 = set(dids_all1)
print('Selected', len(dids_all1), 'documents from ', len(data1))

# dids_all2 = []
# for qid, did_dict in train_pairs2.items():
#     if qid in qids_all:
#         dids_all2.extend(list(did_dict.keys()))
# dids_all2 = set(dids_all2)
# print('Selected', len(dids_all2), 'documents from ', len(data2))

# dids_all = set(list(dids_all1) + list(dids_all2))
dids_all = set(dids_all1)

#
# Finally, we need to add validation query/document IDs to the set of
# queries and documents to save. Otherwise,  respective queries and/or
# documents may be missing.
#

for qid, did_dict in run1.items():
    qids_all.add(qid)
    for did in did_dict.keys():
        dids_all.add(did)

with open_with_default_enc(os.path.join(args.dir_out, QRELS), 'w') as out_qrels_f:
    write_filtered_qrels(out_qrels_f, qrels1, qids_all)
    write_filtered_qrels(out_qrels_f, qrels2, qids_all)

with open_with_default_enc(os.path.join(args.dir_out, DATA_DOCS), 'w') as out_data_f:
    write_filtered_datafiles(out_data_f, data1, DATA_TYPE_DOC, dids_all)
    write_filtered_datafiles(out_data_f, data2, DATA_TYPE_DOC, dids_all)

with open_with_default_enc(os.path.join(args.dir_out, DATA_QUERY), 'w') as out_query_f:
    write_filtered_datafiles(out_query_f, queries1, DATA_TYPE_QUERY, qids_all)
    write_filtered_datafiles(out_query_f, queries2, DATA_TYPE_QUERY, qids_all)


write_filtered_train_pairs(os.path.join(args.dir_out, TRAIN_PAIRS),
                           merge_dict(train_pairs1, train_pairs2),
                           qids_all)