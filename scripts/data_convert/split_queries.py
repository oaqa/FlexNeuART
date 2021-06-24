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
# A script to randomly split queries
# Instead of using this directly, one can use a convenience wrapper shell script split_queries.sh.
#
import sys
import os
import random
import json
import argparse

sys.path.append('.')
from scripts.data_convert.split_queries_args import add_basic_query_split_args, QuerySplitArgumentsBase
from scripts.data_convert.convert_common import read_queries
from scripts.common_eval import read_qrels, qrel_entry2_str
from scripts.config import QUESTION_FILE_JSON, QREL_FILE, DOCID_FIELD
from scripts.data_convert.convert_common import FileWrapper, build_query_id_to_partition


def write_queries_files(queries, query_id_to_partition, dst_dir, partitions_names):
    files = [FileWrapper(os.path.join(dst_dir, name, QUESTION_FILE_JSON), "w")
             for name in partitions_names]

    for query in queries:
        query_id = query[DOCID_FIELD]
        partition_id = query_id_to_partition[query_id]
        files[partition_id].write(json.dumps(query))
        files[partition_id].write('\n')

    for file in files:
        file.close()


def write_qrels_files(qrels, query_id_to_partition, dst_dir, partitions_names):
    files = [FileWrapper(os.path.join(dst_dir, name, QREL_FILE), "w")
             for name in partitions_names]

    for qrel in qrels:
        partition_id = query_id_to_partition[qrel.query_id]
        files[partition_id].write(qrel_entry2_str(qrel))
        files[partition_id].write('\n')

    for file in files:
        file.close()


class QuerySplitArguments(QuerySplitArgumentsBase):
    def __init__(self, raw_args):
        super().__init__(raw_args)

    @property
    def src_dir(self):
        return self.raw_args.src_dir

    @property
    def dst_dir(self):
        return self.raw_args.dst_dir


def main():
    parser = argparse.ArgumentParser(description='Split queries and corresponding QREL files.')
    add_basic_query_split_args(parser)

    parser.add_argument('--src_dir',
                        metavar='input data directory',
                        help='input data directory',
                        type=str, required=True)
    parser.add_argument('--dst_dir',
                        metavar='output data directory',
                        help='output data directory',
                        type=str, required=True)

    args = QuerySplitArguments(parser.parse_args())
    print(args.raw_args)

    print("Start reading input files...")
    src_dir = args.src_dir
    queries = read_queries(os.path.join(src_dir, QUESTION_FILE_JSON))

    query_ids = [data[DOCID_FIELD] for data in queries]

    random.seed(args.seed)
    random.shuffle(query_ids)
    print(f"Shuffled query IDs using sid {args.seed}")

    assert len(query_ids) == len(set(query_ids)), "Non-unique queries ids are forbidden!"
    qrels = read_qrels(os.path.join(src_dir, QREL_FILE))
    print("Done reading input files.")

    sizes = args.partitions_sizes(len(query_ids))
    assert len(sizes) == len(args.partitions_names)
    print("Final partitions sizes:", list(zip(args.partitions_names, sizes)))

    query_id_to_partition = build_query_id_to_partition(query_ids, sizes)

    write_queries_files(queries, query_id_to_partition, args.dst_dir, args.partitions_names)
    write_qrels_files(qrels, query_id_to_partition, args.dst_dir, args.partitions_names)


if __name__ == '__main__':
    main()
