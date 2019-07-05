#!/usr/bin/env python

import sys, os

# We will ignore everything having less than this # of relevant docs per query
MIN_REL_DOC_QTY = 1
# We will ignore everything having less than this # of judgments per query
MIN_JUDGE_DOC_QTY = 5

def readQRELs(inpFile, qrelSet):

  ln = 0

  with open(inpFile, 'r') as f:

    for line in f:
      ln += 1
      line = line.strip()
      if line == '':
        continue

      arr = line.split()
      if len(arr) != 4:
        print('Malformed line %d (expected four space-separated fields) %s' % (ln, line))
        sys.exit(1)

      # query id, documenet id, grade
      qid, _, did, grade = arr

      if not qid in qrelSet:
        qrelSet[qid] = []

      qrelSet[qid].append( (did, int(grade)) )


def writeQRELs(outFile, qrelSet):
  sortedKeys = [int(x) for x, _ in qrelSet.items()]
  sortedKeys.sort()
  sortedKeys = [str(x) for x in sortedKeys]
  with open(outFile, 'w') as f:
    for qid in sortedKeys:
      for did, grade in qrelSet[qid]:
        f.write('%s 0 %s %d\n' % (qid, did, grade))


def readQueries(inpFile):
  res = []
  ln = 0

  with open(inpFile, 'r') as f:

    for line in f:
      ln += 1
      line = line.strip()
      if line == '':
        continue

      arr = line.split(':')
      if len(arr) != 2:
        print('Malformed line %d (expected two :-separated fields) %s' % (ln, line))
        sys.exit(1)

      res.append( tuple(arr) )

  return res

def writeQueries(outFile, tupleArr):

  with open(outFile, 'w') as of:
  
    for qid, text in tupleArr:
    
      of.write(qid + ':' + text + '\n')

def normQueryText(q):
  return q.strip().lower()

# Expecting everything to be in a root dir.
# Source qrels are in qrels.all
#
rootDir = sys.argv[1]

qrels1MQ = {}

readQRELs(os.path.join(rootDir, 'qrels.all', 'qrels1MQ.txt'), qrels1MQ)

writeQRELs(os.path.join(rootDir, 'output', 'train', 'qrels_all_graded.txt'), qrels1MQ)

usedQueries = set()

trecWebQueries = []

qrelsWeb = {}

for year in range(2009, 2013):

  readQRELs(os.path.join(rootDir, 'qrels.all', 'qrels%d.txt' % year), qrelsWeb)

  for qid, qtext in readQueries(os.path.join(rootDir, 'queries.all', 'queries%d.txt' % year)):
    qnorm = normQueryText(qtext)
    usedQueries.add(qnorm)
    trecWebQueries.append( (qid, qnorm) )


writeQRELs(os.path.join(rootDir, 'output', 'test', 'qrels_all_graded.txt'), qrelsWeb)

writeQueries(os.path.join(rootDir, 'output', 'test', 'queries.txt'), trecWebQueries)

trec1MQQueries = []

queryQty = 0

for qid, qtext in readQueries(os.path.join(rootDir, 'queries.all', 'queries1MQ.txt')):
  qnorm = normQueryText(qtext)
  if qnorm in usedQueries:
    print('Ignoring query %s:%s b/c it is present in the regular Web Trec' % (qid, qtext))

  if not qid in qrels1MQ:
    continue # meh no qrels

  queryQrels = qrels1MQ[qid]

  if len(queryQrels) < MIN_JUDGE_DOC_QTY \
    or len(list(filter(lambda x: x[1] > 0, queryQrels))) < MIN_REL_DOC_QTY:
    continue

  trec1MQQueries.append( (qid, qnorm) )

writeQueries(os.path.join(rootDir, 'output', 'train', 'queries.txt'), trec1MQQueries)

print(queryQty)

