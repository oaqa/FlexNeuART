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
import sys
import os
import argparse
import random
import math

sys.path.append('.')
from scripts.data_convert.convert_common import read_queries, write_queries
from scripts.eval_common import read_qrels, write_qrels
from scripts.config import QUESTION_FILE_JSON, QREL_FILE, DOCID_FIELD

parser = argparse.ArgumentParser(description='Sample queries and corresponding QREL entries.')

parser.add_argument('--data_dir',
                    metavar='data directory',
                    help='data directory',
                    type=str, required=True)
parser.add_argument('--input_subdir',
                    metavar='input data subirectory',
                    help='input data subdirectory',
                    type=str, required=True)
parser.add_argument('--seed',
                    metavar='random seed',
                    help='random seed',
                    type=int, default=0)
parser.add_argument('--out_subdir',
                    metavar='output data subirectory',
                    help='output data subdirectory',
                    type=str, required=True)
parser.add_argument('--qty',
                    metavar='1st part # of entries',
                    help='# of entries in the 1st part',
                    type=int, default=None)

args = parser.parse_args()
print(args)

data_dir = args.data_dir

query_id_list = []

query_list = read_queries(os.path.join(data_dir, args.input_subdir, QUESTION_FILE_JSON))

for data in query_list:
    did = data[DOCID_FIELD]
    query_id_list.append(did)

print('Read %d the queries' % (len(query_id_list)))

qrel_list = read_qrels(os.path.join(data_dir, args.input_subdir, QREL_FILE))

print('Read all the QRELs')
# print(qrel_list[0:10])


# print('Before shuffling:', query_id_list[0:10], '...')

random.seed(args.seed)
random.shuffle(query_id_list)
# print('After shuffling:', query_id_list[0:10], '...')

if len(query_id_list) == 0:
    print('Nothing to sample, input is empty')
    sys.exit(1)

sel_query_ids = set(query_id_list[0:args.qty])

print('We selected %d queries' % len(sel_query_ids))

out_dir = os.path.join(data_dir, args.out_subdir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

queries_filtered = list(filter(lambda e: e[DOCID_FIELD] in sel_query_ids, query_list))
qrels_filtered = list(filter(lambda e: e.query_id in sel_query_ids, qrel_list))

write_qrels(qrels_filtered,os.path.join(out_dir, QREL_FILE))
write_queries(queries_filtered, os.path.join(out_dir, QUESTION_FILE_JSON))
