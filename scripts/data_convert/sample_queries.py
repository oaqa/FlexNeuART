#!/usr/bin/env python
import sys
import os
import argparse
import random
import math

sys.path.append('.')
from scripts.data_convert.convert_common import readQueries, writeQueries
from scripts.common_eval import readQrels, writeQrels
from scripts.config import QUESTION_FILE_JSON, QREL_FILE, DOCID_FIELD

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
parser.add_argument('--out_subdir',
                    metavar='output data subirectory',
                    help='output data subdirectory',
                    type=str, required=True)
parser.add_argument('--qty',
                    metavar='1st part # of entries',
                    help='# of entries in the 1st part',
                    type=int, default=None)

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
# print(qrelList[0:10])


# print('Before shuffling:', queryIdList[0:10], '...')

random.seed(args.seed)
random.shuffle(queryIdList)
# print('After shuffling:', queryIdList[0:10], '...')

if len(queryIdList) == 0:
    print('Nothing to sample, input is empty')
    sys.exit(1)

selQueryIds = set(queryIdList[0:args.qty])

print('We selected %d queries' % len(selQueryIds))

outDir = os.path.join(dataDir, args.out_subdir)
if not os.path.exists(outDir):
    os.makedirs(outDir)

queriesFiltered = list(filter(lambda e: e.queryId in selQueryIds, queryList))
qrelsFiltered = list(filter(lambda e: e.queryId in selQueryIds, qrelList))

writeQrels(qrelsFiltered,os.path.join(outDir, QREL_FILE))
writeQueries(queriesFiltered, os.path.join(outDir, QUESTION_FILE_JSON))
