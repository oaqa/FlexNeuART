#!/usr/bin/env python
# Filtering queries to exclude queries that might textually match queries from a set of sub-directories
import sys
import os
import json
import argparse

sys.path.append('.')

from scripts.data_convert.convert_common import FileWrapper, read_queries
from scripts.config import TEXT_FIELD_NAME, QUESTION_FILE_JSON

parser = argparse.ArgumentParser(description='Filter queries to exclude queries from given sub-directories')
parser.add_argument('--input_dir', metavar='input dir', help='input dir',
                    type=str, required=True)
parser.add_argument('--filter_query_dir', metavar='filtering query dir',
                    default=[],
                    help=f'all queries found in {QUESTION_FILE_JSON} files from these directories are ignored',
                    nargs='*')
parser.add_argument('--out_dir', metavar='output directory', help='output directory',
                    type=str, required=True)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

ignore_queries = set()

for qfile_dir in args.filter_query_dir:
    qfile_name = os.path.join(qfile_dir, QUESTION_FILE_JSON)
    for e in read_queries(qfile_name):
        if not TEXT_FIELD_NAME in e:
            continue
        ignore_queries.add(e[TEXT_FIELD_NAME])
    print('Read queries from: ' + qfile_name)

print('A list of queries to ignore has %d entries' % (len(ignore_queries)))

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

out_file_queries = FileWrapper(os.path.join(args.out_dir, QUESTION_FILE_JSON), 'w')

read_qty = 0
wrote_qty = 0

for e in read_queries(os.path.join(args.input_dir, QUESTION_FILE_JSON)):
    read_qty += 1
    if not TEXT_FIELD_NAME in e:
        continue

    text = e[TEXT_FIELD_NAME]
    if text in ignore_queries:
        print(f"Ignoring query, which is found in specified query files: {text}'")
        continue

    wrote_qty += 1
    out_file_queries.write(json.dumps(e) + '\n')


ignored_qty = read_qty - wrote_qty
print(f'Wrote {wrote_qty} queries, ignored {ignored_qty} queries')

out_file_queries.close()

