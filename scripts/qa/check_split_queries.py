#!/usr/bin/env python
import sys
import os
import json
import argparse
import random
import math

sys.path.append('scripts')
from data_convert.convert_common import *

parser = argparse.ArgumentParser(
  description='Checking correctness of split for queries and corresponding QREL files.')

parser.add_argument('--data_dir',
                    metavar='data directory',
                    help='data directory',
                    type=str, required=True)
parser.add_argument('--input_subdir',
                    metavar='input data subirectory',
                    help='input data subdirectory',
                    type=str, required=True)
parser.add_argument('--out_subdir1',
                    metavar='1st output data subirectory',
                    help='1st output data subdirectory',
                    type=str, required=True)
parser.add_argument('--out_subdir2',
                    metavar='2d output data subirectory',
                    help='2d output data subdirectory',
                    type=str, required=True)

args = parser.parse_args()
print(args)



dataDir = args.data_dir

fullQueryList = readQueries(os.path.join(dataDir, args.input_subdir, QUESTION_FILE_JSON))
fullQueryIdSet = set([data[DOCID_FIELD] for data in fullQueryList])

print('Read all the queries from the main dir')

qrelList = readQrels(os.path.join(dataDir, args.input_subdir, QREL_FILE))

print('Read all the QRELs from the main dir')



queryIdSet = set()

partSubDirs = [args.out_subdir1, args.out_subdir2]

for part in range(0,2):
  outDir = os.path.join(dataDir, partSubDirs[part])
  qrelList = readQrels(os.path.join(outDir, QREL_FILE))

  queryPartList = readQueries(os.path.join(outDir, QUESTION_FILE_JSON))
  queryIdPartSet = set([e[DOCID_FIELD] for e in queryPartList])

  queryIdSet = queryIdSet.union(queryIdPartSet)

  # 1. Let's check if any QREL ids have query IDs beyond the current part
  for e in qrelList:
    if e.queryId not in queryIdPartSet:
      print('Qrel entry has query ID not included into %s: %s' %
            (partSubDirs[part], qrelEntry2Str(e)))
      sys.exit(1)

  qrelQueryIdPartSet = set([e.queryId for e in qrelList])
  print('Part %s # of queries # %d of queries with at least one QREL: %d' %
        (partSubDirs[part], len(queryIdPartSet), len(qrelQueryIdPartSet)))

diff = queryIdSet.symmetric_difference(fullQueryIdSet)

print('# of queries in the original folder: %d # of queries in split folders: %d # of queries in the symmetric diff. %d'
      % (len(queryIdSet), len(fullQueryIdSet), len(diff)))

if len(queryIdSet) != len(fullQueryIdSet) or len(diff) > 0:
  print('Query set mismatch!')
  sys.exit(1)

print('Check is successful!')




