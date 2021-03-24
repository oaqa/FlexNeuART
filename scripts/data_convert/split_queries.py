#!/usr/bin/env python
import sys
import os
import argparse
import random
import math

sys.path.append('.')
from scripts.data_convert.convert_common import read_queries, write_queries
from scripts.common_eval import read_qrels, write_qrels
from scripts.config import QUESTION_FILE_JSON, QREL_FILE, DOCID_FIELD

parser = argparse.ArgumentParser(description='Split queries and corresponding QREL files.')

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
parser.add_argument('--out_subdir1',
                    metavar='1st output data subirectory',
                    help='1st output data subdirectory',
                    type=str, required=True)
parser.add_argument('--out_subdir2',
                    metavar='2d output data subirectory',
                    help='2d output data subdirectory',
                    type=str, required=True)
parser.add_argument('--part1_qty',
                    metavar='1st part # of entries',
                    help='# of entries in the 1st part',
                    type=int, default=None)
parser.add_argument('--part1_fract',
                    metavar='1st part fraction # of entries',
                    help='Fraction of entries in the 1st part (from 0 to 1)',
                    type=float, default=None)

args = parser.parse_args()
print(args)

data_dir = args.data_dir

query_id_list = []

query_list = read_queries(os.path.join(data_dir, args.input_subdir, QUESTION_FILE_JSON))

for data in query_list:
    did = data[DOCID_FIELD]
    query_id_list.append(did)

print('Read all the queries')

qrel_list = read_qrels(os.path.join(data_dir, args.input_subdir, QREL_FILE))

print('Read all the QRELs')
# print(qrel_list[0:10])


# print('Before shuffling:', query_id_list[0:10], '...')

random.seed(args.seed)
random.shuffle(query_id_list)

# print('After shuffling:', query_id_list[0:10], '...')

qty = len(query_id_list)

if qty == 0:
    print('Nothing to split, input is empty')
    sys.exit(1)

qty_part = args.part1_qty
if qty_part is None:
    if args.part1_fract is not None:
        if args.part1_fract <= 0 or args.part1_fract >= 1:
            print('The fraction should be > 0 and < 1')
            sys.exit(1)
        qty_part = int(math.ceil(qty * args.part1_fract))
    else:
        print('Specify either --part1_qty or part1_fract')
        sys.exit(1)

query_id_set = set(query_id_list)

qrels_to_ignore = list(filter(lambda e: e.query_id not in query_id_set, qrel_list))

print('# of QRELs with query IDs not present in any part', len(qrels_to_ignore))

sel_query_ids = set(query_id_list[0:qty_part])

print('The first part will have %d documents' % len(sel_query_ids))

part_sub_dirs = [args.out_subdir1, args.out_subdir2]

for part in range(0, 2):
    out_dir = os.path.join(data_dir, part_sub_dirs[part])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    query_part_list = list(filter(lambda e: int(e[DOCID_FIELD] in sel_query_ids) == 1 - part, query_list))
    query_part_id_set = set([e[DOCID_FIELD] for e in query_part_list])

    qrel_part_list = list(filter(lambda e: e.query_id in query_part_id_set, qrel_list))
    write_qrels(qrel_part_list,
               os.path.join(out_dir, QREL_FILE))

    write_queries(query_part_list, os.path.join(out_dir, QUESTION_FILE_JSON))

    print('Part %s # of queries: %d # of QRELs: %d' % (part_sub_dirs[part], len(query_part_list), len(qrel_part_list)))
