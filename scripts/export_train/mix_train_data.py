#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
This scripts mixes two training sets: The validation set is taken from the 1st one.
"""

import argparse
import sys
import os
import numpy as np

from flexneuart.utils import merge_dict
from flexneuart.io import open_with_default_enc

sys.path.append('.')

from flexneuart.io.train_data import QRELS, TRAIN_PAIRS, TEST_RUN, DATA_DOCS, DATA_QUERY, DATA_TYPE_QUERY, DATA_TYPE_DOC
from flexneuart.io.train_data import read_pairs_dict, write_filtered_train_pairs, \
    write_filtered_qrels, read_datafiles, write_filtered_datafiles
from flexneuart.io.runs import read_run_dict, write_run_dict
from flexneuart.io.qrels import read_qrels_dict
from flexneuart.utils import set_all_seeds

def prefix_key(d, pref):
    return {pref + k : v for k, v in d.items()}

def prefix_key_val(d, pref):
    return {pref + k : prefix_key(v, pref) for k, v in d.items()}

parser = argparse.ArgumentParser('Mix multiple training sets in CEDR format (evaluation file is taken from the last one)')

parser.add_argument('--input_dirs', type=str, nargs='+', required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--prob', type=float, nargs='+')

args = parser.parse_args()
print(args)

set_all_seeds(args.seed)

input_dirs = args.input_dirs
output_dir = args.output_dir
input_qty = len(input_dirs)
assert input_qty

print('Input directories:')
for d in input_dirs:
    print(d)
print('=================')
print('Output:', output_dir)

if args.prob:
    probs = np.array(args.prob)
    assert input_qty == len(probs) or len(probs) == 0, \
        'The number of probabilities should match the number of input directories (or do not use probabilities at all)'
    assert np.all(probs > 0), 'probabilities should be > 0'
    assert np.all(probs <= 1), 'probabilities should be <= 1'
    print('Sampling probabilities (used separately for each collection):', probs)
else:
    probs = np.full(1, input_qty)

print('Mixing probabilities:', probs)

os.makedirs(output_dir, exist_ok=True)

with open_with_default_enc(os.path.join(output_dir, QRELS), 'w') as out_qrels_f:
    with open_with_default_enc(os.path.join(output_dir, DATA_DOCS), 'w') as out_data_f:
        with open_with_default_enc(os.path.join(output_dir, DATA_QUERY), 'w') as out_query_f:
            with open_with_default_enc(os.path.join(output_dir, TRAIN_PAIRS), 'w') as out_trainp_f:
                for inp_id, inp_dir in enumerate(input_dirs):
                    out_pref = f'set_{inp_id}'

                    dids = []
                    qids = []

                    # Read convert & save the run
                    if inp_id + 1 >= input_qty:
                        run = prefix_key_val(read_run_dict(os.path.join(inp_dir, TEST_RUN)), out_pref)
                        write_run_dict(run, os.path.join(output_dir, TEST_RUN))

                        # We need to add validation query/document IDs to the set of queries and documents to save.
                        for qid, did_dict in run.items():
                            qids.append(qid)
                            dids.extend(list(did_dict.keys()))

                    queries, data = read_datafiles([os.path.join(inp_dir, fn) for fn in [DATA_DOCS, DATA_QUERY]])
                    queries = prefix_key(queries, out_pref)
                    data = prefix_key(data, out_pref)

                    train_pairs = prefix_key_val(read_pairs_dict(os.path.join(inp_dir, TRAIN_PAIRS)), out_pref)
                    qids.extend(list(train_pairs.keys()))

                    qrels = prefix_key_val(read_qrels_dict(os.path.join(inp_dir, QRELS)), out_pref)

                    prob = probs[inp_id]
                    if prob < 1:
                        qids = list(np.random.choice(qids, size=int(prob * len(qids)), replace=False))

                    print('Set', inp_id, 'Selected ', len(qids), 'training queries from ', len(train_pairs))

                    qids = set(qids)

                    for qid, did_dict in train_pairs.items():
                        if qid in qids:
                            dids.extend(list(did_dict.keys()))
                    dids = set(dids)
                    print('Selected', len(dids), 'documents from ', len(data))

                    write_filtered_qrels(out_qrels_f, qrels, qids)
                    write_filtered_datafiles(out_data_f, data, DATA_TYPE_DOC, dids)
                    write_filtered_datafiles(out_query_f, queries, DATA_TYPE_QUERY, qids)
                    write_filtered_train_pairs(out_trainp_f, train_pairs, qids)



