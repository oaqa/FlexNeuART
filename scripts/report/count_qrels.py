#!/usr/bin/env python
import sys
import argparse

sys.path.append('.')

from scripts.common_eval import read_qrels_dict

parser = argparse.ArgumentParser(description='Count tokens and number of entries in JSONL')

parser.add_argument('--input', type=str, required=True)
parser.add_argument('--min_rel_grade', type=int, default=1)

args = parser.parse_args()

qty = 0
for qid, qrel_dict in read_qrels_dict(args.input).items():
    qty += sum([grade >= args.min_rel_grade for did, grade in qrel_dict.items()])

print(qty)
