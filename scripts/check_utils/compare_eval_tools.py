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
import numpy as np

from flexneuart.io.runs import read_run_dict, write_run_dict
from flexneuart.io.qrels import read_qrels_dict, write_qrels, QrelEntry
from flexneuart.eval import METRIC_LIST, get_eval_results
from flexneuart.io import create_temp_file

parser = argparse.ArgumentParser('Comparing external and internal eval tools')

parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                    type=str, required=True)
parser.add_argument('--run', metavar='a run file', help='a run file',
                    type=str, required=True)
parser.add_argument('--eps', metavar='max relative ratio',
                    help='a threshold to report query-specific differences',
                    type=float, required=True)
parser.add_argument('--eval_metric', choices=METRIC_LIST, default=METRIC_LIST[0],
                    help='Metric list: ' + ','.join(METRIC_LIST),
                    metavar='eval metric')

args = parser.parse_args()
print(args)

qrels = read_qrels_dict(args.qrels)
run = read_run_dict(args.run)
query_ids = list(run.keys())


tmp_file_name_run = create_temp_file()
tmp_file_name_qrels = create_temp_file()

print('query external internal')
print('-----------------------')

res_all_int = []
res_all_ext = []

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
                                  run_file=tmp_file_name_run,
                                  use_qrel_cache=False))

    val_ext, val_int = res[0], res[1]
    mx = max(val_ext, val_int)
    eps = args.eps
    if mx >= eps and abs(val_int - val_ext) / mx >= eps:
        print(qid, val_ext, val_int)
    res_all_ext.append(val_ext)
    res_all_int.append(val_int)

print('-----------------------')
print('mean', np.mean(res_all_ext), np.mean(res_all_int))



