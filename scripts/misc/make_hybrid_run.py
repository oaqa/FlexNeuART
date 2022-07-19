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
    A simple script to merge runs.
    Earlier run has precedence: if a certain result repeats we only keep the entry from the first run.
"""
from flexneuart.io.runs import read_run_dict, write_run_dict, get_sorted_scores_from_score_dict

import argparse


parser = argparse.ArgumentParser(description='Filtering data fields')

parser.add_argument('--input', type=str,
                    required=True,
                    nargs='+', # multiple inputs
                    metavar='input files',
                    help='at least two input files need to be specified (can be gz or bz2 compressed)')
parser.add_argument('--output', type=str,
                    required=True,
                    metavar='output file',
                    help='output run file (can be gz or bz2 compressed)')

args = parser.parse_args()
print(args)

output_run = {}

for inp_f in args.input:
    run = read_run_dict(inp_f)
    for qid, run_dict_1q in run.items():
        for did, score in get_sorted_scores_from_score_dict(run_dict_1q):
            odict = output_run.get(qid, {})
            if did in odict:
                continue
            odict[did] = score
            output_run[qid] = odict

write_run_dict(output_run, args.output)