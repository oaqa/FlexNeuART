#!/usr/bin/env python
# Create a vocabulary from contents of a JSONL input file field
import sys
import argparse

from tqdm import tqdm

sys.path.append('.')

from scripts.data_convert.convert_common import jsonl_gen
from scripts.cedr.data import VocabBuilder

parser = argparse.ArgumentParser('Build vocabularies from several processed fiels')
parser.add_argument('--field_name', metavar='field name', help='a JSONL field to use', required=True)
parser.add_argument('--input', metavar='input files', help='input JSONL files (possibly compressed)',
                    type=str, nargs='+', required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)

args = parser.parse_args()
print(args)

vocab = VocabBuilder()
field = args.field_name

for fn in args.input:
    ln = 0
    for doc_entry in tqdm(jsonl_gen(fn), desc='Processing: ' + fn):
        ln += 1
        if field in doc_entry:
            vocab.proc_doc(doc_entry[field])
        else:
            print(f'WARNING: No field {field} is found in line {ln} file {fn}')
            continue

vocab.save(args.output)
