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
import sys
import argparse

sys.path.append('.')

from scripts.eval_common import read_qrels_dict

parser = argparse.ArgumentParser(description='Count tokens and number of entries in JSONL')

parser.add_argument('--input', type=str, required=True)
parser.add_argument('--min_rel_grade', type=int, default=1)

args = parser.parse_args()

qty = 0
for qid, qrel_dict in read_qrels_dict(args.input).items():
    qty += sum([grade >= args.min_rel_grade for did, grade in qrel_dict.items()])

print(qty)
