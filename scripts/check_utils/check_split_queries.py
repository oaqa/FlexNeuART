#!/usr/bin/env python
#
# Checking correctness of split for queries and corresponding QREL files.
# Basically this is to double-check if the split queries script did the right job.
# Don't use it directly use the wrapper shell script check_split_queries.sh instead.
#
import sys
import os
import argparse

sys.path.append('.')

from scripts.data_convert.convert_common import read_queries
from scripts.common_eval import  read_qrels, qrel_entry2_str
from scripts.config import QUESTION_FILE_JSON, QREL_FILE, DOCID_FIELD

parser = argparse.ArgumentParser(
  description='Checking correctness of split for queries and corresponding QREL files.')

parser.add_argument('--data_dir',
                    metavar='data directory',
                    help='data directory',
                    type=str, required=True)
parser.add_argument('--input_subdir',
                    metavar='input data subirectory',
                    help='input data subdirectory',
                    type=str, required=True)
parser.add_argument('--out_subdir1',
                    metavar='1st output data subirectory',
                    help='1st output data subdirectory',
                    type=str, required=True)
parser.add_argument('--out_subdir2',
                    metavar='2d output data subirectory',
                    help='2d output data subdirectory',
                    type=str, required=True)

args = parser.parse_args()
print(args)



data_dir = args.data_dir

full_query_list = read_queries(os.path.join(data_dir, args.input_subdir, QUESTION_FILE_JSON))
full_query_id_set = set([data[DOCID_FIELD] for data in full_query_list])

print('Read all the queries from the main dir')

qrel_list = read_qrels(os.path.join(data_dir, args.input_subdir, QREL_FILE))

print('Read all the QRELs from the main dir')



query_id_set = set()

part_sub_dirs = [args.out_subdir1, args.out_subdir2]

for part in range(0,2):
  out_dir = os.path.join(data_dir, part_sub_dirs[part])
  qrel_list = read_qrels(os.path.join(out_dir, QREL_FILE))

  query_part_list = read_queries(os.path.join(out_dir, QUESTION_FILE_JSON))
  query_id_part_set = set([e[DOCID_FIELD] for e in query_part_list])

  query_id_set = query_id_set.union(query_id_part_set)

  # 1. Let's check if any QREL ids have query IDs beyond the current part
  for e in qrel_list:
    if e.query_id not in query_id_part_set:
      print('Qrel entry has query ID not included into %s: %s' %
            (part_sub_dirs[part], qrel_entry2_str(e)))
      sys.exit(1)

  qrel_query_id_part_set = set([e.query_id for e in qrel_list])
  print('Part %s # of queries # %d of queries with at least one QREL: %d' %
        (part_sub_dirs[part], len(query_id_part_set), len(qrel_query_id_part_set)))

diff = query_id_set.symmetric_difference(full_query_id_set)

print('# of queries in the original folder: %d # of queries in split folders: %d # of queries in the symmetric diff. %d'
      % (len(query_id_set), len(full_query_id_set), len(diff)))

if len(query_id_set) != len(full_query_id_set) or len(diff) > 0:
  print('Query set mismatch!')
  sys.exit(1)

print('Check is successful!')




