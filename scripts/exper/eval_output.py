#!/usr/bin/env python
import argparse
import sys
import math
import numpy as np
import subprocess as sp


def Usage(err):
    if not err is None:
        print(err)
    print(
        "Usage: <qrel file> <trec-format output file> <optional prefix of report files> <optional label (mandatory if the report prefix is specified>")
    sys.exit(1)


MAP = 'map'
RECIP_RANK = 'recip_rank'
NUM_RET = 'num_ret'
NUM_REL = 'num_rel'
NUM_REL_RET = 'num_rel_ret'
P20 = 'P_20'

NDCG10 = 'ndcg_cut_10'
NDCG20 = 'ndcg_cut_20'
NDCG100 = 'ndcg_cut_100'

ERR20 = 'err20'


def parseTrecEvalResults(lines, metrics):
    prevId = ''
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
        res[qid] = {ERR20: float(val2)}
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

outputGdeval = sp.check_output([gdevalScript, qrelFile, trecOut]).decode('utf-8').split('\n')
resGdeval = parseGdevalResults(outputGdeval)
outputTrecEval = sp.check_output([trecEvalBin,
                                  "-m", "ndcg_cut",
                                  "-m", "official",
                                  "-q", qrelFile, trecOut]).decode('utf-8').replace('\t', ' ').split('\n')
resTrecEval = parseTrecEvalResults(outputTrecEval, set([MAP, NUM_REL, NUM_REL_RET, RECIP_RANK, P20, NDCG10, NDCG20, NDCG100]))

overallNumRel = resTrecEval['all'][NUM_REL]
overallNumRelRet = resTrecEval['all'][NUM_REL_RET]
overallRecall = float(overallNumRelRet) / overallNumRel
overallMAP = resTrecEval['all'][MAP]
overallP20 = resTrecEval['all'][P20]
overallRecipRank = resTrecEval['all'][RECIP_RANK]

overallNDCG10 = resTrecEval['all'][NDCG10]
overallNDCG20 = resTrecEval['all'][NDCG20]
overallNDCG100 = resTrecEval['all'][NDCG100]

overallERR20 = resGdeval['amean'][ERR20]

# valsMAP   =[]
# valsRecipRank   =[]
# valsRecall =[]

queryQty = 0

# May actually delete it at some point, it doesn't do much
# Previously it was used to compute percentiles
for qid, entry in resTrecEval.items():
    if qid == 'all': continue
    queryQty += 1
    # valsMAP.append(entry[MAP])
    # valsRecipRank.append(entry[RECIP_RANK])

    numRel = entry[NUM_REL]
    numRelRet = entry[NUM_REL_RET]

    #
    # Actually, for various reasons qrel may not have relevant answers for a given
    # query, although, this should happen very rarely
    #
    if numRel <= 0:
        print("Warning: No relevant documents for qid=%s numRel=%d" % (qid, numRel))
    # valsRecall.append(numRel/numRelRet)

if len(resTrecEval) != len(resGdeval):
    print("Warning: The number of entries returned by trec_eval and gdeval are different!")

reportText = ""
reportText += "# of queries:    %d" % queryQty
reportText += "\n"
reportText += "NDCG@10:         %f" % overallNDCG10
reportText += "\n"
reportText += "NDCG@20:         %f" % overallNDCG20
reportText += "\n"
reportText += "NDCG@100:        %f" % overallNDCG100
reportText += "\n"
reportText += "ERR@20:          %f" % overallERR20
reportText += "\n"
reportText += "P@20:            %f" % overallP20
reportText += "\n"
reportText += "MAP:             %f" % overallMAP
reportText += "\n"
reportText += "Reciprocal rank: %f" % overallRecipRank
reportText += "\n"
reportText += "Recall:          %f" % overallRecall
reportText += "\n"

sys.stdout.write(reportText)
if outPrefix != '':
    fRep = open(outPrefix + '.rep', 'w')
    fRep.write(reportText)
    fRep.close()
    fTSV = open(outPrefix + '.tsv', 'a')
    fTSV.write(
        "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("Label", "queryQty", "NDCG@10", "NDCG@20", "NDCG@100", "ERR@20", "P@20", "MAP", "MRR", "Recall"))
    fTSV.write("%s\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
    label, queryQty, overallNDCG10, overallNDCG20, overallNDCG100, overallERR20, overallP20, overallMAP, overallRecipRank, overallRecall))
    fTSV.close()

    fTrecEval = open(outPrefix + '.trec_eval', 'w')
    for line in outputTrecEval:
        fTrecEval.write(line.rstrip() + '\n')
    fTrecEval.close()

    fGdeval = open(outPrefix + '.gdeval', 'w')
    for line in outputGdeval:
        fGdeval.write(line.rstrip() + '\n')
    fGdeval.close()
