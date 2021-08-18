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
#  A script to merge to files in BSONL formats. Because it loads everything into memory
#  it is probably not practical to merge data files, but it should be generally fine to merge queries
#  Assumptions:
#   1. Each query has equal number of BSONL entries with matching keys. The order does not matter
#   2. There is no overlap in fields except for the document/query ID field
#
import sys
import argparse
from tqdm import tqdm

sys.path.append('.')

from scripts.data_convert.convert_common import FileWrapper, write_json_to_bin, read_json_from_bin, DOCID_FIELD

parser = argparse.ArgumentParser(description='Merge two files in "BSONL" format.')

parser.add_argument('--input1', metavar='1st input file', type=str, required=True)
parser.add_argument('--input2', metavar='1st input file', type=str, required=True)
parser.add_argument('--output', metavar='output file', type=str, required=True)

args = parser.parse_args()

def load_data(file_name):
    res = {}
    print('Reading:', file_name)
    with FileWrapper(file_name, 'rb') as inp_file:
        while True:
            rec = read_json_from_bin(inp_file)
            if rec is None:
                break
            did = rec[DOCID_FIELD]
            res[did] = rec

    return res


inp1 = load_data(args.input1)
inp2 = load_data(args.input2)

if len(inp1) != len(inp2):
    print('Different number of unique entries: ', str(len(inp1)), ' vs ' + str(len(inp2)))
    sys.exit(1)

with FileWrapper(args.output, 'wb') as out_file:
    for did, e1 in tqdm(inp1.items(), "Merging files"):
        if not did in inp2:
            print(f'Key {did} is present only in the first input file')
            sys.exit(1)

        for k, v in inp2[did].items():
            if k != DOCID_FIELD:
                if k in e1:
                    raise Exception(f'Field name overlap between input files: {k} entry ID: {did}')
                e1[k] = v

        write_json_to_bin(e1, out_file)



