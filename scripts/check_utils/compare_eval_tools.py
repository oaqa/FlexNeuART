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

"""
   
    This script compares output from trec_eval with our reimplementation
    Sample QREL & RUNS can be found in this directory.
   
"""

import argparse
import sys
import numpy as np

from flexneuart.io.runs import read_run_dict, write_run_dict
from flexneuart.io.qrels import read_qrels_dict, write_qrels, QrelEntry
from flexneuart.eval import METRIC_LIST, get_eval_results
from flexneuart.io import create_temp_file

TREC_ROUND=4
# Permit the error only in the last digit
TREC_DIFF_EPS=2.0/10**TREC_ROUND

parser = argparse.ArgumentParser('Comparing external and internal eval tools')

parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                    type=str, required=True)
parser.add_argument('--run', metavar='a run file', help='a run file',
                    type=str, required=True)
parser.add_argument('--max_tol_diff', metavar='max tolerable difference from trec_eval',
                    help='if the metric value diverge from trec_eval by more than this value, report failure',
                    type=float, default=1e-5)
default_metric=METRIC_LIST[0]
parser.add_argument('--eval_metric', choices=METRIC_LIST, default=default_metric,
                    help='Metric list: ' + ', '.join(METRIC_LIST) + ' default: ' + default_metric,
                    metavar='eval metric')

args = parser.parse_args()
print(args)

qrels = read_qrels_dict(args.qrels)
run = read_run_dict(args.run)

print('Comparison statistics')
print('query_id trec_eval ours')
print('-----------------------')

_, res_all_int = get_eval_results(use_external_eval=False,
                                  eval_metric=args.eval_metric,
                                  run=run,
                                  qrels=qrels,
                                  ret_query_vals=True)
_, res_all_ext = get_eval_results(use_external_eval=True,
                                  eval_metric=args.eval_metric,
                                  run=run,
                                  qrels=qrels,
                                  ret_query_vals=True)

if len(res_all_ext) != len(res_all_int):
    print(f'Different number of entries returned, internal: {len(res_all_int)}, external: {len(res_all_ext)}')
    sys.exit(1)

if set(res_all_ext.keys()) != set(res_all_int.keys()):
    print(f'Different query IDs return by external and internal.')
    sys.exit(1)

diverge_qty = 0

for qid in res_all_int:
    val_int = round(res_all_int[qid], TREC_ROUND)
    val_ext = round(res_all_ext[qid], TREC_ROUND)
    if abs(val_int - val_ext) > TREC_DIFF_EPS:
        print(qid, val_ext, val_int)
        diverge_qty +=1

print('-----------------------')
print('mean',
      round(np.mean(list(res_all_ext.values())), TREC_ROUND),
      round(np.mean(list(res_all_int.values())), TREC_ROUND))
if diverge_qty > 0:
    print(f'Failed because {diverge_qty} queries diverged substantially (see divergence statistics)!')
    sys.exit(1)
print('Success: no substantial difference is detected!')




