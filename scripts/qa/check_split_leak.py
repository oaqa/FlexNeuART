#!/usr/bin/env python
import sys
import os
import json
import argparse
import random
import math
import pytorch_pretrained_bert
import nmslib
import time
import numpy as np

sys.path.append('scripts')
from data_convert.convert_common import *

QUERY_BATCH_SIZE=32
QUANTILE_QTY=20

np.random.seed(0)


def jaccard(toks1, toks2):
    set1 = set(toks1)
    set2 = set(toks2)
    totQty = len(set1.union(set2))
    if totQty == 0:
        return 0
    return float(len(set1.intersection(set2))) / totQty


def getTokenIds(text):
    toks = BERT_TOKENIZER.tokenize(text)
    toks = [BERT_TOKENIZER.vocab[t] for t in toks]
    return toks

def toksToStr(arr):
    return ' '.join([str(k) for k in arr])


def strToVect(text):
    return toksToStr(unique(getTokenIds(text)))


BERT_TOKENIZER = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

parser = argparse.ArgumentParser(description='Checking for possible high overlaps among QA pairs.')

parser.add_argument('--data_dir',
                    metavar='data directory',
                    help='data directory',
                    type=str, required=True)
parser.add_argument('--input_subdir1',
                    metavar='1st input subdir',
                    help='1st input data subdirectory',
                    type=str, required=True)
parser.add_argument('--input_subdir2',
                    metavar='1st input subdir',
                    help='1st input data subdirectory',
                    type=str, required=True)
parser.add_argument('--sample_prob1',
                    metavar='1st subdir sample prob',
                    type=float, default=1.0)
parser.add_argument('--sample_prob2',
                    metavar='2d subdir sample prob',
                    type=float, default=1.0)



args = parser.parse_args()
print(args)

dataDir = args.data_dir


qpath1=os.path.join(dataDir, args.input_subdir1, QUESTION_FILE_JSON)
fullQueryList1 = readQueries(qpath1)
sampleQueryList1 = np.random.choice(fullQueryList1,
                                    size=int(len(fullQueryList1) * args.sample_prob1),
                                    replace=False)
print('Read %d queries from %s sampled %d' %
      (len(fullQueryList1), qpath1, len(sampleQueryList1)))


qpath2=os.path.join(dataDir, args.input_subdir2, QUESTION_FILE_JSON)
fullQueryList2 = readQueries(qpath2)
sampleQueryList2 = np.random.choice(fullQueryList2,
                                    size=int(len(fullQueryList2) * args.sample_prob2),
                                    replace=False)
print('Read %d queries from %s sampled %d' %
      (len(fullQueryList2), qpath2, len(sampleQueryList2)))

apath1=os.path.join(dataDir, args.input_subdir1, ANSWER_FILE_JSON)
apath2=os.path.join(dataDir, args.input_subdir2, ANSWER_FILE_JSON)

answDict = {}

index = nmslib.init(method='hnsw',
                    space='jaccard_sparse',
                    data_type=nmslib.DataType.OBJECT_AS_STRING)

for start in tqdm(range(0, len(sampleQueryList2), QUERY_BATCH_SIZE), desc='index. 2d query set'):
    qbatch = []
    for e in sampleQueryList2[start:start + QUERY_BATCH_SIZE]:
        qbatch.append(strToVect(e[TEXT_RAW_FIELD_NAME]))

    if qbatch:
        index.addDataPointBatch(qbatch)
    qbatch = []

print('# of data points to index', len(index))

M = 30
efC = 200

numThreads = 4
indexTimeParams = {'M': M, 'indexThreadQty': numThreads, 'efConstruction': efC, 'post' : 0}

# Create an index
start = time.time()
index.createIndex(indexTimeParams)
end = time.time()
print('Index-time parameters', indexTimeParams)
print('Indexing time = %f' % (end-start))

K = 1
efS = 500
queryTimeParams = {'efSearch': efS}
print('Setting query-time parameters', queryTimeParams)
index.setQueryTimeParams(queryTimeParams)

nbrDists = []

for start in tqdm(range(0, len(sampleQueryList1), QUERY_BATCH_SIZE), desc='query w/ 1st query set'):
    qbatch = []
    for e in sampleQueryList1[start:start + QUERY_BATCH_SIZE]:
        qbatch.append(strToVect(e[TEXT_RAW_FIELD_NAME]))

    if qbatch:
        nbrs = index.knnQueryBatch(qbatch, k=K, num_threads=numThreads)
        for k in range(len(qbatch)):
            nbrDists.append(nbrs[k][1])
    qbatch = []


q=list(np.arange(QUANTILE_QTY + 1) / QUANTILE_QTY) + [0.99]
q.sort()

print('Distribution of distances for k=%d' % K)
dst = np.quantile(nbrDists, q = q)
for k in range(len(q)):
    print('%5.03g' % q[k], '%.03g' % dst[k])


# WHAT'S NEXT:
# 2. FIND DISTRIBUTION OF DISTANCES TO NEAREST NEIGHBORS
# 3. FOR SIMILAR QUERIES CHECK HOW SIMILAR ANSWERS ARE

if False:
    rpath1 = os.path.join(dataDir, args.input_subdir1, QREL_FILE)
    qrelDict1 = readQrelsDict(rpath1)
    print('Read %d qrel sets from %s' % (len(qrelDict1), rpath1))
    rpath2 = os.path.join(dataDir, args.input_subdir2, QREL_FILE)
    qrelDict2 = readQrelsDict(rpath2)
    print('Read %d qrel sets from %s' % (len(qrelDict2), rpath2))

    for fn in [apath1, apath2]:
        qty = 0

        for e in tqdm(jsonlGen(fn), desc='loading answers'):
            qty += 1

            answId = e[DOCID_FIELD]
            answText = e[TEXT_RAW_FIELD_NAME]

            answDict[answId] = answText

        print('Read %d answers from %s' % (qty, fn))


print('Check is successful!')




