#!/usr/bin/env python
import sys
import os
import argparse
import random
import math
import json

sys.path.append('.')
from scripts.data_convert.wikipedia_dpr.split_queries_args import parse_args
from scripts.data_convert.convert_common import readQueries
from scripts.common_eval import readQrels, qrelEntry2Str
from scripts.config import QUESTION_FILE_JSON, QREL_FILE, DOCID_FIELD
from scripts.data_convert.convert_common import FileWrapper


def write_queries_files(queries, query_id_to_partition, dst_dir, partitions_names):
    files = [FileWrapper(os.path.join(dst_dir, name, QUESTION_FILE_JSON), "w") for name in partitions_names]

    for query in queries:
        query_id = query[DOCID_FIELD]
        partition_id = query_id_to_partition[query_id]
        files[partition_id].write(json.dumps(query))
        files[partition_id].write('\n')

    for file in files:
        file.close()

def write_qrels_files(qrels, query_id_to_partition, dst_dir, partitions_names):
    files = [FileWrapper(os.path.join(dst_dir, name, QREL_FILE), "w") for name in partitions_names]

    for qrel in qrels:
        query_id = int(qrel.queryId)
        partition_id = query_id_to_partition[query_id]
        files[partition_id].write(qrelEntry2Str(qrel))
        files[partition_id].write('\n')

    for file in files:
        file.close()

def build_query_id_to_partition(queries_ids, sizes):
    query_id_to_partition = dict()
    count = 0
    p = -1
    for id in queries_ids:
        if count == 0:
            p += 1
            count = sizes[p]
        query_id_to_partition[id] = p
        count -= 1
    return query_id_to_partition

def main():
    args = parse_args()
    print(args.raw_args)

    print("Start reading input files...")
    src_dir = args.src_dir
    queries = readQueries(os.path.join(src_dir, QUESTION_FILE_JSON))
    queries_ids = [data[DOCID_FIELD] for data in queries]
    assert len(queries_ids) == len(set(queries_ids)), "Non-unique queries ids are forbidden!"
    qrels = readQrels(os.path.join(src_dir, QREL_FILE))
    print("Done reading input files.")

    random.seed(args.seed)
    random.shuffle(queries_ids)

    sizes = args.partitions_sizes(len(queries_ids))
    assert len(sizes) == len(args.partitions_names)
    print("Final partitions sizes:", list(zip(args.partitions_names, sizes)))

    query_id_to_partition = build_query_id_to_partition(queries_ids, sizes)

    write_queries_files(queries, query_id_to_partition, args.dst_dir, args.partitions_names)
    write_qrels_files(qrels, query_id_to_partition, args.dst_dir, args.partitions_names)


if __name__ == '__main__':
    main()
