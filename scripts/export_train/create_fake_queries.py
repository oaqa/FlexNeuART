#!/usr/bin/env python
# This script creates fake query data, where each query is replaced
# with
import sys
import os

if len(sys.argv) != 3:
  print('Usage: <directory with exported data> <output fake query file>')
  sys.exit(1)

inpDir = sys.argv[1]
outFile = sys.argv[2]

evalQueries = set()
print('Reading validation queries')
with open(os.path.join(inpDir, 'test_run.txt')) as f:
  for line in f:
    line = line.strip()
    if line:
      fields = line.split('\t')
      evalQueries.add(fields[0])

relDocs = dict()

print('Reading qrels')
with open(os.path.join(inpDir, 'qrels.txt')) as f:
  for line in f:
    line = line.strip()
    if line:
      queryId, _, docId, grade = line.split()
      if int(grade) > 0:
        relDocs[queryId] = docId # Memorize just any one if there are several, we memorize the last entry

docTextDict = dict()

print('Reading docs')
with open(os.path.join(inpDir, 'data_docs.tsv')) as f:
  for line in f:
    line = line.strip()
    if line:
      _, docId, docText = line.split('\t')
      docTextDict[docId] = docText


print('Converting queries:')

with open(outFile, 'w') as of:
  with open(os.path.join(inpDir, 'data_query.tsv')) as f:
    for line in f:
      line = line.strip()
      if line:
        pref, queryId, queryText = line.split('\t')
        if queryId in evalQueries:
          of.write(line + '\n')
        elif queryId in relDocs:
          docId = relDocs[queryId]
          if docId in docTextDict:
            of.write('%s\t%s\t%s\n' % (pref, queryId, docTextDict[docId]))


