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
PRINT_TOO_CLOSE_THRESHOLD=0.9 # We want to inspect answers that are too close

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

def strToNMSLIBVect(text):
    """Converts to a string that can be fed to NMSLIB"""
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
parser.add_argument('-k', metavar='k-NN k',
                    type=int, default=1)
parser.add_argument('--min_jacc', metavar='min jaccard to compare answers',
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

rpath1 = os.path.join(dataDir, args.input_subdir1, QREL_FILE)
qrelDict1 = readQrelsDict(rpath1)
print('Read %d qrel sets from %s' % (len(qrelDict1), rpath1))
rpath2 = os.path.join(dataDir, args.input_subdir2, QREL_FILE)
qrelDict2 = readQrelsDict(rpath2)
print('Read %d qrel sets from %s' % (len(qrelDict2), rpath2))

answDictText = {}

for fn in [apath1, apath2]:
    qty = 0

    for e in tqdm(jsonlGen(fn), desc='loading answers'):
        qty += 1

        answId = e[DOCID_FIELD]
        answText = e[TEXT_RAW_FIELD_NAME]

        answDictText[answId] = answText

    print('Read %d answers from %s' % (qty, fn))


index = nmslib.init(method='hnsw',
                    space='jaccard_sparse',
                    data_type=nmslib.DataType.OBJECT_AS_STRING)

for start in tqdm(range(0, len(sampleQueryList2), QUERY_BATCH_SIZE), desc='index. 2d query set'):
    dbatch = []
    for e in sampleQueryList2[start:start + QUERY_BATCH_SIZE]:
        dbatch.append(strToNMSLIBVect(e[TEXT_RAW_FIELD_NAME]))

    if dbatch:
        index.addDataPointBatch(dbatch)
        dbatch = []

print('# of data points to index', len(index))

M = 30
efC = 200

indexTimeParams = {'M': M, 'efConstruction': efC, 'post' : 0}

# Create an index
start = time.time()
index.createIndex(indexTimeParams)
end = time.time()
print('Index-time parameters', indexTimeParams)
print('Indexing time = %f' % (end-start))

K = args.k
print('K=', K)
efS = 1000
queryTimeParams = {'efSearch': efS}
print('Setting query-time parameters', queryTimeParams)
index.setQueryTimeParams(queryTimeParams)

nbrQuestDists = []
nbrAnswDists = []

for start in tqdm(range(0, len(sampleQueryList1), QUERY_BATCH_SIZE), desc='query w/ 1st query set'):
    qbatch = []
    for e in sampleQueryList1[start:start + QUERY_BATCH_SIZE]:
        qbatch.append(strToNMSLIBVect(e[TEXT_RAW_FIELD_NAME]))

    if qbatch:
        nbrs = index.knnQueryBatch(qbatch, k=K, num_threads=numThreads)
        assert(len(nbrs))
        for i in range(len(qbatch)):
            qnum1 = start + i
            qid1 = sampleQueryList1[qnum1][DOCID_FIELD]

            indexQueries, dists = nbrs[i]
            for t in range(len(indexQueries)):
                dist = dists[t]
                nbrQuestDists.append(dist)

                # For close enough queries, compute all pairwise distances
                # between the respective relevant answers
                if dist >= args.min_jacc:
                    qnum2 = indexQueries[t]

                    qid2 = sampleQueryList2[qnum2][DOCID_FIELD]

                    if qid1 in qrelDict1 and qid2 in qrelDict2:
                        for aid1, grade1 in qrelDict1[qid1].items():
                            for aid2, grade2 in qrelDict2[qid2].items():
                                if grade1 > 0 and grade2 > 0 and \
                                    aid1 in answDictText and aid2 in answDictText:
                                    toks1 = unique(getTokenIds(answDictText[aid1]))
                                    toks2 = unique(getTokenIds(answDictText[aid2]))
                                    answDist = jaccard(toks1, toks2)
                                    nbrAnswDists.append(answDist)
                                    if answDist >= PRINT_TOO_CLOSE_THRESHOLD:
                                        print(qid1, aid1, '<=>', answDist, '<=>', qid2, aid2)
                                        print('---------------------')
                                        print(answDictText[aid1])
                                        print(toks1)
                                        print('---------------------')
                                        print(answDictText[aid2])
                                        print(toks2)
                                        print('=====================')

        qbatch = []


q=list(np.arange(QUANTILE_QTY + 1) / QUANTILE_QTY) + [0.99]
q.sort()

print('Distribution of question-neighbor distances for k=%d' % K)
dst = np.quantile(nbrQuestDists, q = q)
for k in range(len(q)):
    print('%5.03g' % q[k], '%.03g' % dst[k])

print('Distribution of relevant answer distances from neighbor questions with Jaccard >= %g' % args.min_jacc)
if nbrAnswDists:
    dst = np.quantile(nbrAnswDists, q = q)
    for k in range(len(q)):
        print('%5.03g' % q[k], '%.03g' % dst[k])
else:
    print('No data collected, did you set the Jaccard threshold to a value < 1?')


print('Check is successful!')




