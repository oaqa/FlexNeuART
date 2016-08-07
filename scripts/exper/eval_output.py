#!/usr/bin/env python
import argparse
import sys
import math
import numpy as np
import subprocess as sp

def Usage(err):
  if not err is None:
    print err
  print "Usage: <trec binary> <qrel file> <trec-format output file> <optional prefix of report files> <optional label (mandatory if the report prefix is specified>"
  sys.exit(1)

RECIP_RANK='recip_rank'
NUM_RET='num_ret'
NUM_REL='num_rel'
NUM_REL_RET='num_rel_ret'
metrics = set([RECIP_RANK, NUM_RET, NUM_REL, NUM_REL_RET])

def readResults(lines):
  prevId=''
  res=dict()
  for s in lines:
    if s == '': continue
    arr=s.split()
    if (len(arr) != 3):
      raise Exception("wrong-format line: '%s'" % s)
    (metr, qid, val) = arr
    if not qid in res: res[qid]=dict()
    entry=res[qid]
    if metr in metrics:
      entry[metr]=float(val)
  return res
  

if len(sys.argv) != 4 and len(sys.argv) != 5 and len(sys.argv) != 6:
  Usage(None)

trecEvalBin=sys.argv[1]
qrelFile=sys.argv[2]
trecOut=sys.argv[3]
outPrefix=''
label=''
if len(sys.argv) >= 5: 
  outPrefix = sys.argv[4]
  if len(sys.argv) != 6: Usage("Specify the 6th arg")
  label=sys.argv[5]
output=sp.check_output([trecEvalBin, "-q", qrelFile, trecOut]).replace('\t', ' ').split('\n')
res=readResults(output)

numRel=res['all'][NUM_REL]
numRelRet=res['all'][NUM_REL_RET]
mrr=res['all'][RECIP_RANK]
gotCorrect=0
recipRanks=[]
recipRanksNonZeroRecall=[]
ranksNonZeroRecall=[]
accsNonZeroRecall=[]
queryQty=0
for qid, entry in res.iteritems():
  if qid == 'all': continue
  queryQty+=1
  recipRanks.append(entry[RECIP_RANK])
  if int(entry[NUM_REL_RET]) == 1:
    rrank=entry[RECIP_RANK]
    recipRanksNonZeroRecall.append(rrank)
    ranksNonZeroRecall.append(1.0/rrank)
    val=0
    if (entry[RECIP_RANK]>=0.999): val=1
    accsNonZeroRecall.append(val)

gotRight=np.sum(accsNonZeroRecall)
precision_at_1_nonzbinrecall=gotRight/numRelRet
precision_at_1_overall=float(gotRight)/queryQty
mrr_nonzbinrecall=np.average(recipRanksNonZeroRecall)


mrrOverall=res['all'][RECIP_RANK]
recall=numRelRet/numRel
reportText=""
reportText += "recall:          %f" % recall
reportText += "\n"
reportText += "# of queries:    %d" % queryQty
reportText += "\n"
reportText += "got correct:     %d" % int(gotRight)
reportText += "\n"
reportText += "p@1:             %f%s" % (precision_at_1_nonzbinrecall, " (for non-zero binary recall)")
reportText += "\n"
reportText += "p@1:             %f%s" % (precision_at_1_overall, " (overall)")
reportText += "\n"
reportText += "MRR:             %f%s" % (mrr_nonzbinrecall, " (for non-zero binary recall)")
reportText += "\n"
reportText += "MRR:             %f%s" % (mrr, " (overall)")
reportText += "\n"
pcts=np.percentile(ranksNonZeroRecall, [50, 75, 90, 99])
reportText += "Rank percentiles 50=%f, 75=%f, 90=%f, 99=%f %s" % (pcts[0], pcts[1], pcts[2], pcts[3], " (for non-zero binary recall)")
reportText += "\n"
sys.stdout.write(reportText)
if outPrefix != '':
  fRep=open(outPrefix +'.rep','w')
  fRep.write(reportText)
  fRep.close()
  fTSV=open(outPrefix +'.tsv','a')
  fTSV.write("%s\t%f\t%d\t%d\t%f\t%f\n" % (label, recall, queryQty, int(gotRight), precision_at_1_nonzbinrecall, mrr_nonzbinrecall))
  fTSV.close()
  fTrecEval=open(outPrefix +'.trec_eval','w')
  for line in output:
    fTrecEval.write(line.rstrip() + '\n')
  fTrecEval.close()
