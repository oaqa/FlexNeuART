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

"""
    Just a simple script to extract a list of of values from a specific field of a JSONL file
"""
import argparse

from flexneuart.io.utils import jsonl_gen

parser = argparse.ArgumentParser('Extract question text')

parser.add_argument('--input', metavar='input JSONL', required=True)
parser.add_argument('--output', metavar='output text', required=True)
parser.add_argument('--field_name', metavar='field name', required=True)

args = parser.parse_args()

fn = args.field_name

with open(args.output, 'w') as out_f:
    for e in jsonl_gen(args.input):
        if fn in e:
            text = e[fn]
            if text is not None:
                text = text.strip()
            if text:
                out_f.write(text + '\n')
