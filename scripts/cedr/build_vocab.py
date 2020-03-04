#!/usr/bin/env python
import sys
import argparse
from data import *

sys.path.append('scripts')
from data_convert.convert_common import *


parser = argparse.ArgumentParser('Build vocabularies from several processed fiels')
parser.add_argument('--field', metavar='field', help='a JSON field to use', required=True)
parser.add_argument('--input', metavar='input files', help='input JSON files (possibly compressed)',
                    type=str, nargs='+', required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)

args = parser.parse_args()
print(args)


vocab=VocabBuilder()
field = args.field

for fn in args.input:
  for docEntry in tqdm(jsonlGen(fn), desc='Processing: ' + fn):
    if field in docEntry:
      vocab.procDoc(docEntry[field])


vocab.save(args.output)


