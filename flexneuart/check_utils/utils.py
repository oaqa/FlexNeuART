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
import numpy as np
import nmslib
import time

from tqdm import tqdm

from flexneuart.data_convert.utils import unique
from flexneuart.config import QUESTION_FILE_JSON, TEXT_RAW_FIELD_NAME
from flexneuart.io.queries import read_queries

QUERY_BATCH_SIZE=32


def jaccard(toks1, toks2):
    set1 = set(toks1)
    set2 = set(toks2)
    tot_qty = len(set1.union(set2))
    if tot_qty == 0:
        return 0
    return float(len(set1.intersection(set2))) / tot_qty


def get_token_ids(tokenizer, text):
    toks = tokenizer.tokenize(text)
    toks = [tokenizer.vocab[t] for t in toks]
    return toks


def toks_to_str(arr):
    return ' '.join([str(k) for k in arr])


def str_to_nmslib_vect(tokenizer, text):
    """Converts to a string that can be fed to NMSLIB"""
    lst = unique(get_token_ids(tokenizer, text))
    lst.sort()
    return toks_to_str(lst)


def read_sample_queries(data_dir,
                      input_subdir1, sample_prob1,
                      input_subdir2, sample_prob2):
    qpath1 = os.path.join(data_dir, input_subdir1, QUESTION_FILE_JSON)
    full_query_list1 = read_queries(qpath1)
    sample_query_list1 = np.random.choice(full_query_list1,
                                        size=int(len(full_query_list1) * sample_prob1),
                                        replace=False)
    print('Read %d queries from %s sampled %d' %
          (len(full_query_list1), qpath1, len(sample_query_list1)))

    qpath2 = os.path.join(data_dir, input_subdir2, QUESTION_FILE_JSON)
    full_query_list2 = read_queries(qpath2)
    sample_query_list2 = np.random.choice(full_query_list2,
                                        size=int(len(full_query_list2) * sample_prob2),
                                        replace=False)
    print('Read %d queries from %s sampled %d' %
          (len(full_query_list2), qpath2, len(sample_query_list2)))

    return sample_query_list1, sample_query_list2


def create_jaccard_index(use_hnsw, tokenizer, query_list):
    if use_hnsw:
        method_name = 'hnsw'
        M = 30
        ef_c = 200

        index_time_params = {'M': M, 'efConstruction': ef_c, 'indexThreadQty': 0, 'post': 0}
        ef_s = 1000
        query_time_params = {'efSearch': ef_s}
    else:
        method_name = 'brute_force'
        index_time_params = {}
        query_time_params = {}

    print('k-NN search method', method_name)

    index = nmslib.init(method=method_name,
                        space='jaccard_sparse',
                        data_type=nmslib.DataType.OBJECT_AS_STRING)

    for start in tqdm(range(0, len(query_list), QUERY_BATCH_SIZE), desc='reading query set to build a k-NN search index'):
        dbatch = []
        for e in query_list[start:start + QUERY_BATCH_SIZE]:
            dbatch.append(str_to_nmslib_vect(tokenizer, e[TEXT_RAW_FIELD_NAME]))

        if dbatch:
            index.addDataPointBatch(dbatch)
            dbatch = []

    print('# of data points to index', len(index))

    # Create an index
    start = time.time()
    index.createIndex(index_time_params)
    end = time.time()
    print('Index-time parameters', index_time_params)
    print('Indexing time = %f' % (end - start))

    print('Setting query-time parameters', query_time_params)
    index.setQueryTimeParams(query_time_params)

    return index
