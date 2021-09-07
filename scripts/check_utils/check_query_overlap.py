#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import argparse
import numpy as np
from tqdm import tqdm

#
# This script checks for possibly overlapping queries in different folders.
# First, it checks for exact match with respect to tokenized text.
# Second, it checks for approximate (high-enough) match with respect to BERT tokens
#

from scripts.check_utils.check_common import get_token_ids, str_to_nmslib_vect, jaccard, read_sample_queries, create_jaccard_index
from scripts.config import get_bert_tokenizer, DOCID_FIELD, TEXT_FIELD_NAME, TEXT_RAW_FIELD_NAME

QUERY_BATCH_SIZE=32

np.random.seed(0)

tokenizer = get_bert_tokenizer()

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

data_dir = args.data_dir

data_dir = args.data_dir

sample_query_list1, sample_query_list2 = read_sample_queries(data_dir,
                                                       args.input_subdir1, args.sample_prob1,
                                                       args.input_subdir2, args.sample_prob2)

# Making sample_query_list2 a longer one
if len(sample_query_list2) < len(sample_query_list1):
    prev_sample_query_list2 = sample_query_list2
    sample_query_list2 = sample_query_list1
    sample_query_list1 = prev_sample_query_list2


seen_text = set([e[TEXT_FIELD_NAME].strip() for e in sample_query_list2])

rep_tok_qty = 0
for e in sample_query_list1:
    text = e[TEXT_FIELD_NAME].strip()
    if text and text in seen_text:
        print(f'Repeating tokenized query, id: {e[DOCID_FIELD]}, text: {text}')
        rep_tok_qty += 1

index = create_jaccard_index(args.use_hnsw, tokenizer, sample_query_list2)

rep_approx_qty = 0
for start in tqdm(range(0, len(sample_query_list1), QUERY_BATCH_SIZE), desc='query w/ 1st query set'):
    qbatch = []
    for e in sample_query_list1[start:start + QUERY_BATCH_SIZE]:
        qbatch.append(str_to_nmslib_vect(tokenizer, e[TEXT_RAW_FIELD_NAME]))

    if qbatch:
        nbrs = index.knnQueryBatch(qbatch, k=1, num_threads=0)
        assert(len(nbrs))
        for i in range(len(qbatch)):
            qnum1 = start + i
            qid1 = sample_query_list1[qnum1][DOCID_FIELD]
            raw_text = sample_query_list1[qnum1][TEXT_RAW_FIELD_NAME]

            index_queries, dists = nbrs[i]
            # In the case of Jaccard, the similarity is one minus the distance
            if len(index_queries) > 0:
                qsim = 1 - dists[0]
                if qsim >= args.min_jacc:
                    qnum2 = index_queries[0]
                    qid2 = sample_query_list2[qnum2][DOCID_FIELD]
                    raw_text2 = sample_query_list2[qnum2][TEXT_RAW_FIELD_NAME]
                    toks1 = get_token_ids(tokenizer, raw_text)
                    toks2 = get_token_ids(tokenizer, raw_text2)
                    qsim = jaccard(toks1, toks2)
                    if qsim >= args.min_jacc:
                        rep_approx_qty += 1
                        print(f'Approximate match between id: {qid1} and id: {qid2}, Jaccard score: {qsim}')
                        print(f'1st query: {raw_text}')
                        print(f'2d query: {raw_text2}')



print(f'# of exact tokenized matches: {rep_tok_qty} # of approx. matches: {rep_approx_qty}')



