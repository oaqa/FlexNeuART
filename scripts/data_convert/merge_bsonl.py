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

parser.add_argument('--input', metavar='1st input file', type=str,
                    nargs='+', required=True)
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


input_arr = []
for file_id in range(len(args.input)):
    input_arr.append(load_data(args.input[file_id]))
    if file_id > 0 and len(input_arr[file_id]) != len(input_arr[file_id-1]):
        print('Different number of unique entries: ',
              str(len(input_arr[file_id])), ' vs ' + str(len(input_arr[file_id-1])),
              'for files ', args.input[file_id], 'and', args.input[file_id-1])
        sys.exit(1)


with FileWrapper(args.output, 'wb') as out_file:
    inp0 = input_arr[0]
    fn0 = args.input[0]
    for did, e0 in tqdm(inp0.items(), "Merging files"):
        for file_id in range(1, len(input_arr)):
            inp1 = input_arr[file_id]
            fn1 = args.input[file_id]
            if not did in inp1:
                print(f'Key {did} is present only in {inp0} but not in {inp1}')
                sys.exit(1)

            for field_name, field_val in inp1[did].items():
                if field_name != DOCID_FIELD:
                    if field_name in e0:
                        raise Exception(f'Field name overlap between input files:  entry ID: {did}')
                    e0[field_name] = field_val

        write_json_to_bin(e0, out_file)



