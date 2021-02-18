#!/usr/bin/env python
# A simple script to remove unnecessary fields from an input file (to make leaner and smaller).
# The document ID field will not be removed.

import sys
import json
import argparse

sys.path.append('.')

from scripts.config import DOCID_FIELD
from scripts.data_convert.convert_common import jsonlGen, FileWrapper

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
    for ln, old_rec in enumerate(jsonlGen(args.input)):
        if DOCID_FIELD not in old_rec:
            raise Exception(f'Entry {ln+1} in args.input lacks the field {DOCID_FIELD}')
        new_rec = {k : old_rec[k] for k in set(old_rec.keys()).intersection(incl_field_set)}
        fout.write(json.dumps(new_rec) + '\n')