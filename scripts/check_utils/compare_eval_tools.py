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
query_ids = list(run.keys())

tmp_file_name_run = create_temp_file()
tmp_file_name_qrels = create_temp_file()

print('Comparison statistics')
print('query_id trec_eval ours')
print('-----------------------')

res_all_int = []
res_all_ext = []

diverge_qty = 0

for qid in query_ids:
    tmp_qrels = []
    if qid in qrels:
        for did, rel_grade in qrels[qid].items():
            tmp_qrels.append(QrelEntry(query_id=qid, doc_id=did, rel_grade=rel_grade))
    tmp_run = {qid : run[qid]}
    write_qrels(tmp_qrels, tmp_file_name_qrels)
    write_run_dict(tmp_run, tmp_file_name_run)
    res = []
    for i in range(2):
        res.append(get_eval_results(i==0,
                                  args.eval_metric,
                                  rerank_run=tmp_run,
                                  qrel_file=tmp_file_name_qrels,
                                  run_file=tmp_file_name_run))

    val_ext, val_int = res[0], round(res[1], TREC_ROUND)
    if abs(val_int - val_ext) > TREC_DIFF_EPS:
        print(qid, val_ext, val_int)
        diverge_qty +=1
    res_all_ext.append(val_ext)
    res_all_int.append(val_int)

print('-----------------------')
print('mean', round(np.mean(res_all_ext), TREC_ROUND), round(np.mean(res_all_int), TREC_ROUND))
if diverge_qty > 0:
    print(f'Failed because {diverge_qty} queries diverged substantially (see divergence statistics)!')
    sys.exit(1)
print('Success: no substantial difference is detected!')




