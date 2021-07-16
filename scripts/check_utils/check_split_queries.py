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
#
# Checking correctness of the split for queries and corresponding QREL files.
# Basically this is to double-check if the split queries script did the right job.
# Instead of using this directly, one can use a convenience wrapper shell script check_split_queries.sh.
#
import sys
import os
import argparse

sys.path.append('.')

from scripts.data_convert.convert_common import read_queries
from scripts.common_eval import read_qrels, qrel_entry2_str
from scripts.config import QUESTION_FILE_JSON, QREL_FILE, DOCID_FIELD

parser = argparse.ArgumentParser(
    description='Checking correctness of split for queries and corresponding QREL files.')

parser.add_argument('--src_dir',
                    metavar='input data directory',
                    help='input data directory',
                    type=str, required=True)
parser.add_argument('--dst_dir',
                    metavar='output data directory',
                    help='output data directory',
                    type=str, required=True)
parser.add_argument('--partitions_names',
                    metavar='names of partitions to split at',
                    help='names of partitions to split at separated by comma',
                    required=True,
                    type=str)

args = parser.parse_args()
print(args)

dst_dir = args.dst_dir

full_query_list = read_queries(os.path.join(args.src_dir, QUESTION_FILE_JSON))
full_query_id_set = set([data[DOCID_FIELD] for data in full_query_list])

print('Read all the queries from the main dir')

qrel_list = read_qrels(os.path.join(args.src_dir, QREL_FILE))

print('Read all the QRELs from the main dir')

query_id_set = set()

part_sub_dirs = args.partitions_names.split(',')

for part_id in range(len(part_sub_dirs)):
    out_dir = os.path.join(dst_dir, part_sub_dirs[part_id])
    qrel_list = read_qrels(os.path.join(out_dir, QREL_FILE))

    query_part_list = read_queries(os.path.join(out_dir, QUESTION_FILE_JSON))
    query_id_part_set = set([e[DOCID_FIELD] for e in query_part_list])

    query_id_set = query_id_set.union(query_id_part_set)

    # 1. Let's check if any QREL ids have query IDs beyond the current part
    for e in qrel_list:
        if e.query_id not in query_id_part_set:
            print('Qrel entry has query ID not included into %s: %s' %
                  (part_sub_dirs[part_id], qrel_entry2_str(e)))
            sys.exit(1)

    qrel_query_id_part_set = set([e.query_id for e in qrel_list])
    print('Part %s # of queries # %d of queries with at least one QREL: %d' %
          (part_sub_dirs[part_id], len(query_id_part_set), len(qrel_query_id_part_set)))

diff = query_id_set.symmetric_difference(full_query_id_set)

print('# of queries in the original folder: %d # of queries in split folders: %d # of queries in the symmetric diff. %d'
      % (len(full_query_id_set), len(query_id_set), len(diff)))

if len(query_id_set) != len(full_query_id_set) or len(diff) > 0:
    print('Query set mismatch!')
    sys.exit(1)

print('Check is successful!')
