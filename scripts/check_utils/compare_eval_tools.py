#!/usr/bin/env python
# This script compares output from trec_eval with our reimplementation
# Sample QREL & RUNS can be found in this directory.
import sys
import argparse
import tempfile
import os

sys.path.append('.')

from scripts.common_eval import *
from scripts.utils import create_temp_file

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

qrels = readQrelsDict(args.qrels)
run = readRunDict(args.run)
query_ids = list(run.keys())


tmpFileNameRun = create_temp_file()
tmpFileNameQrels = create_temp_file()

print('query external internal')
print('-----------------------')

res_all_int = []
res_all_ext = []

for qid in query_ids:
    tmpQrels = []
    if qid in qrels:
        for did, relGrade in qrels[qid].items():
            tmpQrels.append(QrelEntry(queryId=qid, docId=did, relGrade=relGrade))
    tmpRun = {qid : run[qid]}
    writeQrels(tmpQrels, tmpFileNameQrels)
    writeRunDict(tmpRun, tmpFileNameRun)
    res = []
    for i in range(2):
        res.append(getEvalResults(i==0,
                                  args.eval_metric,
                                  rerankRun=tmpRun,
                                  qrelFile=tmpFileNameQrels,
                                  runFile=tmpFileNameRun,
                                  useQrelCache=False))

    val_ext, val_int = res[0], res[1]
    mx = max(val_ext, val_int)
    eps = args.eps
    if mx >= eps and abs(val_int - val_ext) / mx >= eps:
        print(qid, val_ext, val_int)
    res_all_ext.append(val_ext)
    res_all_int.append(val_int)

print('-----------------------')
print('mean', np.mean(res_all_ext), np.mean(res_all_int))



