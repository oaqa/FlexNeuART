#!/usr/bin/env python
import sys

def fatalError(msg):
  print >> sys.stderr, msg
  exit(1)

inpFile=sys.argv[1]
outFile=sys.argv[2]
minRecallAt1=float(sys.argv[3])
verbose=len(sys.argv)==5 and int(sys.argv[4])>0
if verbose: print "Input file: %s min recall@1: %f" % (inpFile, minRecallAt1)

f = open(inpFile)
fo = open(outFile, 'a')

s = f.readline().strip()
head = s.split('\t')
if len(head) < 3:
  fatalError("Wrong header: two few fields")
if head[0] != 'MethodName':
  fatalError("Wrong header: the first field is %s instead of MethodName" % head[0])
recallAt1Id=None
recallId=None
indexParamId=None
queryTimeParamId=None
indexImprEff=None
indexImprDistComp=None
indexQueryTime=None
for i in range(0,len(head)):
  if head[i] == 'Recall@1': recallAt1Id=i
  if head[i] == 'Recall': recallId=i
  if head[i] == 'IndexParams': indexParamId=i
  if head[i] == 'ImprEfficiency': indexImprEff=i
  if head[i] == 'ImprDistComp': indexImprDistComp=i
  if head[i] == 'QueryTimeParams': queryTimeParamId=i
  if head[i] == 'QueryTime': indexQueryTime=i

if recallAt1Id is None:
  fatalError("wrong header, no Recall@1")
if recallId is None:
  fatalError("wrong header, no Recall")
if indexParamId is None:
  fatalError("wrong header, no IndexParams")
if indexImprEff is None:
  fatalError("wrong header, no ImprEfficiency")
if indexImprDistComp is None:
  fatalError("wrong header, no ImprDistComp")
if queryTimeParamId is None:
  fatalError("wrong header, no QueryTimeParams")

if verbose: print "Recall@1 field id: %d IndexParams field id %d" % (recallAt1Id, indexParamId)

bestImprEff=-1
bestImprDistComp=None
bestIndexTimeParams=None
bestQueryTimeParams=None
bestRecallAt1=None
bestRecall=None
bestTime=None

for s in f.readlines():
  fields=s.strip().split('\t')
  recallAt1=float(fields[recallAt1Id])
  recall=float(fields[recallId])
  imprEff = float(fields[indexImprEff])
  if recallAt1 >= minRecallAt1 and imprEff > bestImprEff:
    bestImprEff=imprEff
    bestRecallAt1=recallAt1
    bestRecall=recall
    bestIndexTimeParams=fields[indexParamId]
    bestImprDistComp=float(fields[indexImprDistComp])
    bestQueryTimeParams=fields[queryTimeParamId]
    bestTime=float(fields[indexQueryTime])

if bestIndexTimeParams is None:
  fatalError("No data found for the minimum recall@1=%f" % minRecallAt1)
else:
  if verbose:
    print "Best improvement in efficiency %f time %f (impr. dist comp=%f recall@1=%f recall=%f) params: index-time=%s query-time=%s" % (bestImprEff,bestTime,bestImprDistComp,bestRecallAt1,bestRecall,bestIndexTimeParams,bestQueryTimeParams);
    print bestIndexTimeParams, bestQueryTimeParams
    # Note the space before \ !
    fo.write("%s %s \\\n" % (bestIndexTimeParams, bestQueryTimeParams))
  else:
    print "%f\t%f\t%f" %(bestImprDistComp, bestRecallAt1, bestRecall)
  print ""
