#!/usr/bin/env python
# Filtering queries to exclude queries that might textually match queries from a set of sub-directories
import sys
import os
import json
import argparse

sys.path.append('.')

from scripts.data_convert.convert_common import FileWrapper, readQueries
from scripts.config import TEXT_FIELD_NAME, QUESTION_FILE_JSON

parser = argparse.ArgumentParser(description='Filter queries to exclude queries from given sub-directories')
parser.add_argument('--input', metavar='input file', help='input file',
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

ignoreQueries = set()

for qfile_dir in args.filter_query_dir:
    qfile_name = os.path.join(qfile_dir, QUESTION_FILE_JSON)
    for e in readQueries(qfile_name):
        if not TEXT_FIELD_NAME in e:
            continue
        ignoreQueries.add(e[TEXT_FIELD_NAME])
    print('Read queries from: ' + qfile_name)

print('A list of queries to ignore has %d entries' % (len(ignoreQueries)))

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

outFileQueries = FileWrapper(os.path.join(args.out_dir, QUESTION_FILE_JSON), 'w')

readQty = 0
wroteQty = 0

for e in readQueries(args.input):
    readQty += 1
    if not TEXT_FIELD_NAME in e:
        continue

    text = e[TEXT_FIELD_NAME]
    if text in ignoreQueries:
        print(f"Ignoring query, which is found in specified query files: {text}'")

    wroteQty += 1
    outFileQueries.write(json.dumps(e) + '\n')


ignoredQty = readQty - wroteQty
print(f'Wrote {wroteQty} queries, ignored {ignoredQty} queries')

outFileQueries.close()

