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
import subprocess as sp

# This was just to check compatibiity with some old runs
RUN_GDEVAL=False

def Usage(err):
    if not err is None:
        print(err)
    print(
        "Usage: <qrel file> <trec-format output file> <optional prefix of report files> <optional label (mandatory if the report prefix is specified>")
    sys.exit(1)


NUM_REL = 'num_rel'
NUM_REL_RET = 'num_rel_ret'

GDEVAL_ERR20= 'err20'
P20         = 'P_20'

MAP         = 'map'
RECIP_RANK  = 'recip_rank'
RECALL      = 'recall'

GDEVAL_NDCG20           = 'ndcg20'
GDEVAL_NDCG20_REPORT    = 'GDEVAL NDCG@20'

METRIC_DICT = {
    MAP             : 'MAP',
    RECIP_RANK      : 'MRR',
    P20             : 'P20',
    RECALL          : 'Recall',

    GDEVAL_ERR20    : 'ERR@20',
    GDEVAL_NDCG20   :  GDEVAL_NDCG20_REPORT
}

FINAL_METR_ORDERED_LIST=[]

for k in [10, 20, 100]:
    nkey = f'ndcg_cut_{k}'
    METRIC_DICT[nkey] = f'NDCG@{k}'
    FINAL_METR_ORDERED_LIST.append(nkey)

FINAL_METR_ORDERED_LIST.extend([P20, MAP, RECIP_RANK, RECALL])
if RUN_GDEVAL:
    FINAL_METR_ORDERED_LIST.extend([GDEVAL_ERR20, GDEVAL_NDCG20])

# Recall is computed from NUM_REL and NUM_REL_RET
TREC_EVAL_METR = [k for k in METRIC_DICT.keys() if k not in[RECALL]]
NUM_REL_TREC_EVAL_METRICS = [NUM_REL, NUM_REL_RET]
TREC_EVAL_METR.extend(NUM_REL_TREC_EVAL_METRICS)


def parse_trec_eval_results(lines, metrics):
    metrics = set(metrics)
    res = dict()
    for s in lines:
        if s == '': continue
        arr = s.split()
        if (len(arr) != 3):
            raise Exception("wrong-format line: '%s'" % s)
        (metr, qid, val) = arr
        if not qid in res: res[qid] = dict()
        entry = res[qid]
        if metr in metrics:
            entry[metr] = float(val)
    return res


def parse_gdeval_results(lines):
    res = dict()
    first = True
    for s in lines:
        if s == '': continue
        if first:
            first = False
            continue
        arr = s.split(',')
        if (len(arr) != 4):
            raise Exception("wrong-format line: '%s'" % s)
        (runid, qid, val1, val2) = arr
        res[qid] = {GDEVAL_NDCG20: float(val1), GDEVAL_ERR20: float(val2)}
    return res


if len(sys.argv) != 3 and len(sys.argv) != 5:
    Usage(None)

trec_eval_bin = 'trec_eval/trec_eval'
gdeval_script = 'scripts/exper/gdeval.pl'

qrel_file = sys.argv[1]
trec_out = sys.argv[2]
out_prefix = ''
label = ''
if len(sys.argv) >= 4:
    out_prefix = sys.argv[3]
    if len(sys.argv) != 5: Usage("Specify the 4th arg")
    label = sys.argv[4]

output_trec_eval = sp.check_output([trec_eval_bin,
                                  "-m", "ndcg_cut",
                                  "-m", "official",
                                  "-q", qrel_file, trec_out]).decode('utf-8').replace('\t', ' ').split('\n')

res_trec_eval_all = parse_trec_eval_results(output_trec_eval, TREC_EVAL_METR)
res_trec_eval=res_trec_eval_all['all']

#print('trec_eval results parsed:', res_trec_eval)

# Some manipulations are required for these metrics
res = {RECALL : float(res_trec_eval[NUM_REL_RET]) / res_trec_eval[NUM_REL]}

# Just "pass-through" metric with results coming directly from trec_eval
for k in FINAL_METR_ORDERED_LIST:
    if k in res_trec_eval:
        res[k] = res_trec_eval[k]

if RUN_GDEVAL:
    output_gdeval = sp.check_output([gdeval_script, qrel_file, trec_out]).decode('utf-8').split('\n')
    res_gdeval_all = parse_gdeval_results(output_gdeval)
    res_gdeval=res_gdeval_all['amean']

#print('gdeval results parsed:', res_gdeval)

if RUN_GDEVAL:
    res[GDEVAL_ERR20] = res_gdeval[GDEVAL_ERR20]
    res[GDEVAL_NDCG20] = res_gdeval[GDEVAL_NDCG20]

query_qty = 0

# Previously it was used to compute percentiles,
# currently it just prints a warning and computes the number of queries
for qid, entry in res_trec_eval_all.items():
    if qid == 'all': continue
    query_qty += 1

    num_rel = entry[NUM_REL]

    if num_rel <= 0:
        print("Warning: No relevant documents for qid=%s num_rel=%d" % (qid, num_rel))

if RUN_GDEVAL:
    if len(res_trec_eval_all) != len(res_gdeval_all):
        print("Warning: The number of query entries returned by trec_eval and gdeval are different!")

report_text = f"# of queries:    {query_qty}\n"

maxl = 0
for k in FINAL_METR_ORDERED_LIST:
    maxl = max(len(METRIC_DICT[k]), maxl)

for k in FINAL_METR_ORDERED_LIST:
    name = METRIC_DICT[k] + ': ' + ''.join([' '] * (maxl - len(METRIC_DICT[k])))
    report_text += (name + '%f') % res[k] + '\n'

sys.stdout.write(report_text)
if out_prefix != '':
    f_rep = open(out_prefix + '.rep', 'w')
    f_rep.write(report_text)
    f_rep.close()
    f_tsv = open(out_prefix + '.tsv', 'a')

    header = ["Label", "queryQty"]
    data = [label, str(query_qty)]

    for k in FINAL_METR_ORDERED_LIST:
        header.append(METRIC_DICT[k])
        data.append('%f' % res[k])

    f_tsv.write('\t'.join(header) + '\n')
    f_tsv.write('\t'.join(data) + '\n')

    f_tsv.close()

    f_trec_eval = open(out_prefix + '.trec_eval', 'w')
    for line in output_trec_eval:
        f_trec_eval.write(line.rstrip() + '\n')
    f_trec_eval.close()

    if RUN_GDEVAL:
        f_gdeval = open(out_prefix + '.gdeval', 'w')
        for line in output_gdeval:
            f_gdeval.write(line.rstrip() + '\n')
        f_gdeval.close()
