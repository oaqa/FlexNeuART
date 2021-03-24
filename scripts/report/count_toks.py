#!/usr/bin/env python
import sys
import argparse

sys.path.append('.')

from scripts.data_convert.convert_common import jsonl_gen

parser = argparse.ArgumentParser(description='Count tokens and number of entries in JSONL')

parser.add_argument('--input', type=str, required=True)
parser.add_argument('--field', type=str, required=True)

args = parser.parse_args()

qty = 0
tok_qty = 0
field = args.field

for e in jsonl_gen(args.input):
    qty += 1
    if field in e:
        tok_qty += len(e[field].split())


print(qty, tok_qty)