#!/usr/bin/env python
import sys
import subprocess as sp


def Usage(err):
    if not err is None:
        print(err)
    print(
        "Usage: <qrel file> <trec-format output file> <optional prefix of report files> <optional label (mandatory if the report prefix is specified>")
    sys.exit(1)


NUM_REL = 'num_rel'
NUM_REL_RET = 'num_rel_ret'

ERR20       = 'err20'
P20         = 'P_20'

MAP         = 'map'
RECIP_RANK  = 'recip_rank'
RECALL      = 'recall'

GDEVAL_NDCG20           = 'ndcg20'
GDEVAL_NDCG20_REPORT    = 'GDEVAL NDCG@20'

METRIC_DICT = {
    MAP             : 'MAP',
    RECIP_RANK      : 'MRR',
    ERR20           : 'ERR@20',
    P20             : 'P20',
    RECALL          : 'Recall',
    GDEVAL_NDCG20   :  GDEVAL_NDCG20_REPORT
}

FINAL_METR_ORDERED_LIST=[]

for k in [10, 20, 100]:
    nkey = f'ndcg_cut_{k}'
    METRIC_DICT[nkey] = f'NDCG@{k}'
    FINAL_METR_ORDERED_LIST.append(nkey)

FINAL_METR_ORDERED_LIST.extend([ERR20, P20, MAP, RECIP_RANK, RECALL, GDEVAL_NDCG20])

# Recall is computed from NUM_REL and NUM_REL_RET
TREC_EVAL_METR = [k for k in METRIC_DICT.keys() if k not in[RECALL]]
NUM_REL_TREC_EVAL_METRICS = [NUM_REL, NUM_REL_RET]
TREC_EVAL_METR.extend(NUM_REL_TREC_EVAL_METRICS)


def parseTrecEvalResults(lines, metrics):
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


def parseGdevalResults(lines):
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
        res[qid] = {GDEVAL_NDCG20: float(val1), ERR20: float(val2)}
    return res


if len(sys.argv) != 3 and len(sys.argv) != 5:
    Usage(None)

trecEvalBin = 'trec_eval/trec_eval'
gdevalScript = 'scripts/exper/gdeval.pl'

qrelFile = sys.argv[1]
trecOut = sys.argv[2]
outPrefix = ''
label = ''
if len(sys.argv) >= 4:
    outPrefix = sys.argv[3]
    if len(sys.argv) != 5: Usage("Specify the 4th arg")
    label = sys.argv[4]

outputTrecEval = sp.check_output([trecEvalBin,
                                  "-m", "ndcg_cut",
                                  "-m", "official",
                                  "-q", qrelFile, trecOut]).decode('utf-8').replace('\t', ' ').split('\n')

resTrecEvalAll = parseTrecEvalResults(outputTrecEval, TREC_EVAL_METR)
resTrecEval=resTrecEvalAll['all']

#print('trec_eval results parsed:', resTrecEval)

# Some manipulations are required for these metrics
res = {RECALL : float(resTrecEval[NUM_REL_RET]) / resTrecEval[NUM_REL]}

# Just "pass-through" metric with results coming directly from trec_eval
for k in FINAL_METR_ORDERED_LIST:
    if k in resTrecEval:
        res[k] = resTrecEval[k]

outputGdeval = sp.check_output([gdevalScript, qrelFile, trecOut]).decode('utf-8').split('\n')
resGdevalAll = parseGdevalResults(outputGdeval)
resGdeval=resGdevalAll['amean']

#print('gdeval results parsed:', resGdeval)

res[ERR20] = resGdeval[ERR20]
res[GDEVAL_NDCG20] = resGdeval[GDEVAL_NDCG20]

queryQty = 0

# Previously it was used to compute percentiles,
# currently it just prints a warning and computes the number of queries
for qid, entry in resTrecEvalAll.items():
    if qid == 'all': continue
    queryQty += 1

    numRel = entry[NUM_REL]

    if numRel <= 0:
        print("Warning: No relevant documents for qid=%s numRel=%d" % (qid, numRel))


if len(resTrecEvalAll) != len(resGdevalAll):
    print("Warning: The number of query entries returned by trec_eval and gdeval are different!")

reportText = f"# of queries:    {queryQty}\n"

maxl = 0
for k in FINAL_METR_ORDERED_LIST:
    maxl = max(len(METRIC_DICT[k]), maxl)

for k in FINAL_METR_ORDERED_LIST:
    name = METRIC_DICT[k] + ': ' + ''.join([' '] * (maxl - len(METRIC_DICT[k])))
    reportText += (name + '%f') % res[k] + '\n'

sys.stdout.write(reportText)
if outPrefix != '':
    fRep = open(outPrefix + '.rep', 'w')
    fRep.write(reportText)
    fRep.close()
    fTSV = open(outPrefix + '.tsv', 'a')

    header = ["Label", "queryQty"]
    data = [label, str(queryQty)]

    for k in FINAL_METR_ORDERED_LIST:
        header.append(METRIC_DICT[k])
        data.append('%f' % res[k])

    fTSV.write('\t'.join(header) + '\n')
    fTSV.write('\t'.join(data) + '\n')

    fTSV.close()

    fTrecEval = open(outPrefix + '.trec_eval', 'w')
    for line in outputTrecEval:
        fTrecEval.write(line.rstrip() + '\n')
    fTrecEval.close()

    fGdeval = open(outPrefix + '.gdeval', 'w')
    for line in outputGdeval:
        fGdeval.write(line.rstrip() + '\n')
    fGdeval.close()
