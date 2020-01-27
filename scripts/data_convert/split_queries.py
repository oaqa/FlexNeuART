#!/usr/bin/env python
import sys
import os
import json
import argparse
import random
import math

sys.path.append('scripts')
from data_convert.convert_common import *

parser = argparse.ArgumentParser(description='Split queries and corresponding QREL files.')

parser.add_argument('--data_dir',
                    metavar='data directory',
                    help='data directory',
                    type=str, required=True)
parser.add_argument('--input_subdir',
                    metavar='input data subirectory',
                    help='input data subdirectory',
                    type=str, required=True)
parser.add_argument('--seed',
                    metavar='random seed',
                    help='random seed',
                    type=int, default=0)
parser.add_argument('--out_subdir1',
                    metavar='1st output data subirectory',
                    help='1st output data subdirectory',
                    type=str, required=True)
parser.add_argument('--out_subdir2',
                    metavar='2d output data subirectory',
                    help='2d output data subdirectory',
                    type=str, required=True)
parser.add_argument('--part1_qty',
                    metavar='1st part # of entries',
                    help='# of entries in the 1st part',
                    type=int, default=None)
parser.add_argument('--part1_fract',
                    metavar='1st part fraction # of entries',
                    help='Fraction of entries in the 1st part (from 0 to 1)',
                    type=float, default=None)

args = parser.parse_args()
print(args)

dataDir = args.data_dir

queryIdList = []

queryList = readQueries(os.path.join(dataDir, args.input_subdir, QUESTION_FILE_JSON))

for data in queryList:
  did = data[DOCID_FIELD]
  queryIdList.append(did)

print('Read all the queries')

qrelList = readQrels(os.path.join(dataDir, args.input_subdir, QREL_FILE))

print('Read all the QRELs')
#print(qrelList[0:10])


#print('Before shuffling:', queryIdList[0:10], '...')

random.seed(args.seed)
random.shuffle(queryIdList)

#print('After shuffling:', queryIdList[0:10], '...')

qty = len(queryIdList)

if qty == 0:
  print('Nothing to split, input is empty')
  sys.exit(1)

qtyPart = args.part1_qty
if qtyPart is None:
  if args.part1_fract is not None:
    if args.part1_fract <= 0 or args.part1_fract >= 1:
      print('The fraction should be > 0 and < 1')
      sys.exit(1)
    qtyPart = int(math.ceil(qty * args.part1_fract))
  else:
    print('Specify either --part1_qty or part1_fract')
    sys.exit(1)

queryIdSet = set(queryIdList)

qrelsToIgnore = list(filter(lambda e: e.queryId not in queryIdSet, qrelList))

print('# of QRELs with query IDs not present in any part', len(qrelsToIgnore))


selQueryIds = set(queryIdList[0:qtyPart])

print('The first part will have %d documents' % len(selQueryIds))

partSubDirs = [args.out_subdir1, args.out_subdir2]

for part in range(0,2):
  outDir = os.path.join(dataDir, partSubDirs[part])
  if not os.path.exists(outDir):
    os.makedirs(outDir)

  queryPartList = list(filter(lambda e: int(e[DOCID_FIELD] in selQueryIds) == 1 - part, queryList))
  queryPartIdSet = set([e[DOCID_FIELD] for e in queryPartList])

  qrelPartList = list(filter(lambda e: e.queryId in queryPartIdSet, qrelList))
  writeQrels(qrelPartList,
             os.path.join(outDir, QREL_FILE))

  writeQueries(queryPartList, os.path.join(outDir, QUESTION_FILE_JSON))

  print('Part %s # of queries: %d # of QRELs: %d' % (partSubDirs[part], len(queryPartList), len(qrelPartList)))



