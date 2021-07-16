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
# A simple script to remove unnecessary fields from an input file (to make it leaner and smaller).
# The document ID field will not be removed.

import sys
import json
import argparse

sys.path.append('.')

from scripts.config import DOCID_FIELD
from scripts.data_convert.convert_common import jsonl_gen, FileWrapper

parser = argparse.ArgumentParser(description='Filtering data fields')

parser.add_argument('--input', type=str,
                    required=True,
                    metavar='input file',
                    help='input JSONL file (can be gz or bz2 compressed)')
parser.add_argument('--output', type=str,
                    required=True,
                    metavar='output file',
                    help='output JSONL file (can be gz or bz2 compressed)')
parser.add_argument('--keep_fields', nargs='+',
                    metavar='included fields',
                    required=True,
                    help=f'A list of fields to include, note that {DOCID_FIELD} is not filtered out.')


args = parser.parse_args()
print(args)

incl_field_set = set(args.keep_fields + [DOCID_FIELD])

with FileWrapper(args.output, 'w') as fout:
    for ln, old_rec in enumerate(jsonl_gen(args.input)):
        if DOCID_FIELD not in old_rec:
            raise Exception(f'Entry {ln+1} in args.input lacks the field {DOCID_FIELD}')
        new_rec = {k : old_rec[k] for k in set(old_rec.keys()).intersection(incl_field_set)}
        fout.write(json.dumps(new_rec) + '\n')