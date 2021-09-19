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
import os
import argparse
import numpy as np
from tqdm import tqdm

"""
   
    This utility scripts checks for possible leakage across different data splits.
    Importantly it works only for bitext. In the context of a community QA collection,
    such bitext arises naturally. For regular document collections, a pseudo-bitext
    needs to be created user the scripts/giza/export_bitext_plain.sh:
    importantly one needs to use the text_raw field (and text as an index field)
    and use 0 for the  "max query to doc word ratio" so documents are not split
    into chunks.
   

    Specifically, we search for very similar question-answer pairs, which might
    be duplicates or near duplicates. Hence, we check the following:
    1. Are there very similar questions?
    2. For sufficiently similar questions, e.g., Jaccard >= 0.75, we check
       all pairwise similarities among all relevant answers.
   
    By default this method uses brute-force search with the Jaccard similarity.
    The exhaustiveness of the search ensures we won't miss anything. However, for quicker-and-easier
    checks, one can use HNSW with sufficently high values of M (>= 30), efConstruction (>=200),
    and efSearch (>=1000). These parameters might need to be bumped up for "harder" collections
    and brute-force search is certainly a safer option.
   
"""

from flexneuart.check_utils import get_token_ids, QUERY_BATCH_SIZE, jaccard, \
                                   read_sample_queries, create_jaccard_index, str_to_nmslib_vect

from flexneuart.text_proc.parse import get_bert_tokenizer
from flexneuart.io import jsonl_gen
from flexneuart.data_convert import unique
from flexneuart.config import ANSWER_FILE_JSON, QREL_FILE, DOCID_FIELD, TEXT_RAW_FIELD_NAME

from flexneuart.eval import read_qrels_dict

PRINT_TOO_CLOSE_THRESHOLD=0.9 # We want to inspect answers that are too close

np.random.seed(0)

tokenizer = get_bert_tokenizer()

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
parser.add_argument("--use_hnsw", action="store_true",
                    help="Use HNSW instead of brute-force for retrieval")



args = parser.parse_args()
print(args)

data_dir = args.data_dir

sample_query_list1, sample_query_list2 = read_sample_queries(data_dir,
                                                       args.input_subdir1, args.sample_prob1,
                                                       args.input_subdir2, args.sample_prob2)

apath1=os.path.join(data_dir, args.input_subdir1, ANSWER_FILE_JSON)
apath2=os.path.join(data_dir, args.input_subdir2, ANSWER_FILE_JSON)

rpath1 = os.path.join(data_dir, args.input_subdir1, QREL_FILE)
qrel_dict1 = read_qrels_dict(rpath1)
print('Read %d qrel sets from %s' % (len(qrel_dict1), rpath1))
rpath2 = os.path.join(data_dir, args.input_subdir2, QREL_FILE)
qrel_dict2 = read_qrels_dict(rpath2)
print('Read %d qrel sets from %s' % (len(qrel_dict2), rpath2))

answ_dict_text = {}

for fn in [apath1, apath2]:
    qty = 0

    for e in tqdm(jsonl_gen(fn), desc='loading answers'):
        qty += 1

        answ_id = e[DOCID_FIELD]
        answ_text = e[TEXT_RAW_FIELD_NAME]

        answ_dict_text[answ_id] = answ_text

    print('Read %d answers from %s' % (qty, fn))

index = create_jaccard_index(args.use_hnsw, tokenizer, sample_query_list2)

K = args.k
print('K=', K)


nbr_quest_simils = []
nbr_answ_simils = []

for start in tqdm(range(0, len(sample_query_list1), QUERY_BATCH_SIZE), desc='query w/ 1st query set'):
    qbatch = []
    for e in sample_query_list1[start:start + QUERY_BATCH_SIZE]:
        qbatch.append(str_to_nmslib_vect(tokenizer, e[TEXT_RAW_FIELD_NAME]))

    if qbatch:
        nbrs = index.knnQueryBatch(qbatch, k=K, num_threads=0)
        assert(len(nbrs))
        for i in range(len(qbatch)):
            qnum1 = start + i
            qid1 = sample_query_list1[qnum1][DOCID_FIELD]

            index_queries, dists = nbrs[i]
            for t in range(len(index_queries)):
                # In the case of Jaccard, the similarity is one minus the distance
                nqsimil = 1 - dists[t]
                nbr_quest_simils.append(nqsimil)

                # For close enough queries, compute all pairwise distances
                # between the respective relevant answers
                if nqsimil >= args.min_jacc:
                    qnum2 = index_queries[t]

                    qid2 = sample_query_list2[qnum2][DOCID_FIELD]

                    if qid1 in qrel_dict1 and qid2 in qrel_dict2:
                        for aid1, grade1 in qrel_dict1[qid1].items():
                            for aid2, grade2 in qrel_dict2[qid2].items():
                                if grade1 > 0 and grade2 > 0 and \
                                    aid1 in answ_dict_text and aid2 in answ_dict_text:
                                    toks1 = unique(get_token_ids(tokenizer, answ_dict_text[aid1]))
                                    toks2 = unique(get_token_ids(tokenizer, answ_dict_text[aid2]))
                                    answ_simil = jaccard(toks1, toks2)
                                    nbr_answ_simils.append(answ_simil)
                                    if answ_simil >= PRINT_TOO_CLOSE_THRESHOLD:
                                        print(qid1, aid1, '<=>', answ_simil, '<=>', qid2, aid2)
                                        print('---------------------')
                                        print(answ_dict_text[aid1])
                                        print(toks1)
                                        print('---------------------')
                                        print(answ_dict_text[aid2])
                                        print(toks2)
                                        print('=====================')

        qbatch = []

# We are more interested in extremely high similarities, hence,
# we increase resolution in higher quantiles
q=list([0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.999])
q.sort()

print('Maximum similarity among questions:', np.max(nbr_quest_simils))
print('Distribution of question-neighbor *SIMILARITIES* for k=%d' % K)
dst = np.quantile(nbr_quest_simils, q = q)
print(' quant| simil')
print('------+------')
for k in range(len(q)):
    print('%5.03g' % q[k], ' | %.05g' % dst[k])

print('Distribution of relevant answer pairwise *SIMILARITIES* from neighbor questions with Jaccard >= %g' % args.min_jacc)
if nbr_answ_simils:
    dst = np.quantile(nbr_answ_simils, q = q)
    print(' quant| simil')
    print('------+------')
    for k in range(len(q)):
        print('%5.03g' % q[k], ' | %.05g' % dst[k])
else:
    print('No data collected, did you set the Jaccard threshold to a value < 1?')


print('Check is successful!')




