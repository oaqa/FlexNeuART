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
from tqdm import tqdm

# This script checks for possibly overlapping queries in different folders.
# First, it checks for exact match with respect to tokenized text.
# Second, it checks for approximate (high-enough) match with respect to BERT tokens

sys.path.append('.')

from scripts.check_utils.common_check import getTokenIds, strToNMSLIBVect, jaccard, readSampleQueries, createJaccardIndex
from scripts.config import BERT_BASE_MODEL, \
                            DOCID_FIELD, \
                            TEXT_FIELD_NAME, TEXT_RAW_FIELD_NAME

from scripts.common_eval import readQrelsDict

QUERY_BATCH_SIZE=32

np.random.seed(0)


BERT_TOKENIZER = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

parser = argparse.ArgumentParser(description='Checking for possible overlapping queries.')

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
parser.add_argument('--min_jacc', metavar='min jaccard to consider queries to be semi-duplicates',
                    type=float, default=0.95)
parser.add_argument("--use_hnsw",
                    action="store_true",
                    help="Use HNSW instead of brute-force for retrieval")



args = parser.parse_args()
print(args)

dataDir = args.data_dir

dataDir = args.data_dir

sampleQueryList1, sampleQueryList2 = readSampleQueries(dataDir,
                                                       args.input_subdir1, args.sample_prob1,
                                                       args.input_subdir2, args.sample_prob2)

# Making sampleQueryList2 a longer one
if len(sampleQueryList2) < len(sampleQueryList1):
    prevSampleQueryList2 = sampleQueryList2
    sampleQueryList2 = sampleQueryList1
    sampleQueryList1 = prevSampleQueryList2


seenText = set([e[TEXT_FIELD_NAME].join(' ').strip() for e in sampleQueryList2])

repTokQty = 0
for e in sampleQueryList1:
    text = e[TEXT_FIELD_NAME].join(' ').strip()
    if text and text in seenText:
        print(f'Repeating tokenized query, id: {e[DOCID_FIELD]}, text: {text}')
        repTokQty += 1

index = createJaccardIndex(args.use_hnsw, BERT_TOKENIZER, sampleQueryList2)

repApproxQty = 0
for start in tqdm(range(0, len(sampleQueryList1), QUERY_BATCH_SIZE), desc='query w/ 1st query set'):
    qbatch = []
    for e in sampleQueryList1[start:start + QUERY_BATCH_SIZE]:
        qbatch.append(strToNMSLIBVect(BERT_TOKENIZER, e[TEXT_RAW_FIELD_NAME]))

    if qbatch:
        nbrs = index.knnQueryBatch(qbatch, k=1, num_threads=0)
        assert(len(nbrs))
        for i in range(len(qbatch)):
            qnum1 = start + i
            qid1 = sampleQueryList1[qnum1][DOCID_FIELD]
            rawText = sampleQueryList1[qnum1][TEXT_RAW_FIELD_NAME]

            indexQueries, dists = nbrs[i]
            # In the case of Jaccard, the similarity is one minus the distance
            if indexQueries:
                qsim = 1 - dists[0]
                if qsim >= args.min_jacc:
                    repApproxQty += 1
                    qnum2 = indexQueries[0]
                    qid2 = sampleQueryList2[qnum2][DOCID_FIELD]
                    rawText2 = sampleQueryList2[qnum2][TEXT_RAW_FIELD_NAME]
                    print(f'Approximate match between id: {qid1} and id: {qid2}, Jaccard score: {qsim}')
                    print(f'1st query: {rawText}')
                    print(f'2d query: {rawText2}')


print(f'# of exact tokenized matches: {repTokQty} # of approx. matches: {repApproxQty}')



