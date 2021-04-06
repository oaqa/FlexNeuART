#!/usr/bin/env python
# A script to randomly split queries
# Instead of using this directly, one can use a convenience wrapper shell script split_queries.sh.
#
import sys
import os
import random
import json

sys.path.append('.')
from scripts.data_convert.split_queries_args import parse_args
from scripts.data_convert.convert_common import read_queries
from scripts.common_eval import read_qrels, qrel_entry2_str
from scripts.config import QUESTION_FILE_JSON, QREL_FILE, DOCID_FIELD
from scripts.data_convert.convert_common import FileWrapper


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


def build_query_id_to_partition(query_ids, sizes):
    assert sum(sizes) == len(query_ids)
    query_id_to_partition = dict()
    start = 0
    for part_id in range(len(sizes)):
        end = start + sizes[part_id]
        for k in range(start, end):
            query_id_to_partition[query_ids[k]] = part_id
        start = end

    return query_id_to_partition


def main():
    args = parse_args()
    print(args.raw_args)


    random.seed(args.seed)

    print("Start reading input files...")
    src_dir = args.src_dir
    queries = read_queries(os.path.join(src_dir, QUESTION_FILE_JSON))

    print(f"Shuffled query IDs using sid {args.seed}")
    query_ids = [data[DOCID_FIELD] for data in queries]

    random.shuffle(query_ids)

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
