#!/usr/bin/env python
# Just a simple script to extract a list of of questions from the JSONL file
import sys
import argparse

sys.path.append('.')

from scripts.data_convert.convert_common import jsonlGen

parser = argparse.ArgumentParser('Extract question text')

parser.add_argument('--input', metavar='input JSONL', required=True)
parser.add_argument('--output', metavar='output text', required=True)
parser.add_argument('--field_name', metavar='field name', required=True)

args = parser.parse_args()

fn = args.field_name

with open(args.output, 'w') as out_f:
    for e in jsonlGen(args.input):
        if fn in e:
            text = e[fn]
            if text is not None:
                text = text.strip()
            if text:
                out_f.write(text + '\n')
