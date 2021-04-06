#!/usr/bin/env python
# A script to randomly split queries
# Instead of using this directly, one can use a convenience wrapper shell script split_queries.sh.
#
import sys
import os
import tqdm
import random
import argparse

sys.path.append('.')
from scripts.data_convert.split_queries_args import QuerySplitArgumentsBase, add_basic_query_split_args
from scripts.data_convert.convert_common import FileWrapper, build_query_id_to_partition
from scripts.data_convert.wikipedia_dpr.utils import dpr_json_reader


class QuerySplitArguments(QuerySplitArgumentsBase):
    def __init__(self, raw_args):
        super().__init__(raw_args)

    @property
    def src_file(self):
        return self.raw_args.src_file

    @property
    def dst_file_pref(self):
        return self.raw_args.dst_file_pref



def main():
    parser = argparse.ArgumentParser(description='Split raw DPR queries.')
    add_basic_query_split_args(parser)

    parser.add_argument('--src_file',
                        metavar='input file name',
                        help='input file name',
                        type=str, required=True)
    parser.add_argument('--dst_file_pref',
                        metavar='output file prefix',
                        help='output file prefix',
                        type=str, required=True)

    args = QuerySplitArguments(parser.parse_args())

    print(args.raw_args)

    print("Start reading input files...")
    src_file = args.src_file

    query_ids = []

    # First time we read the input file to count the number of queries
    with FileWrapper(src_file) as inp_file:
        for query_idx, _ in tqdm.tqdm(enumerate(dpr_json_reader(inp_file))):
            query_ids.append(query_idx)

    random.seed(args.seed)
    random.shuffle(query_ids)

    print(f"Shuffled query IDs using sid {args.seed}")

    sizes = args.partitions_sizes(len(query_ids))
    assert len(sizes) == len(args.partitions_names)
    print("Final partitions sizes:", list(zip(args.partitions_names, sizes)))

    query_id_to_partition = build_query_id_to_partition(query_ids, sizes)

    out_file_list = [None] * len(args.partitions_names)
    max_query_idx = [-1] * len(args.partitions_names)

    for part_id, part_name in enumerate(args.partitions_names):
        out_file_name = args.dst_file_pref + '_' + part_name + '.json.gz'
        out_file_list[part_id] = FileWrapper(out_file_name, 'w')
        out_file_list[part_id].write('[\n')

    # Due to specifics of formatting of the DPR files, we need to put comma
    # right after the } that "finalizes" a question.
    # However, the last } in the file shouldn't be followed by a comma.
    # To implement this, we need to know the maximum query ID in a partition
    for query_id, part_id in query_id_to_partition.items():
        max_query_idx[part_id] = max(max_query_idx[part_id], query_id)

    # First time we read the input file to actually split things
    with FileWrapper(src_file) as inp_file:
        for query_idx, json_str in tqdm.tqdm(enumerate(dpr_json_reader(inp_file))):
            part_id = query_id_to_partition[query_idx]
            out_file = out_file_list[part_id]
            if query_idx < max_query_idx[part_id]:
                out_file.write(json_str + ',\n')
            else:
                # Final entry shouldn't be followed by a comma
                out_file.write(json_str + '\n')


    for out_file in out_file_list:
        out_file.write(']\n')
        out_file.close()


if __name__ == '__main__':
    main()
