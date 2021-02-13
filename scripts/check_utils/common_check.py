import os
import numpy as np
import nmslib
import time

from tqdm import tqdm

from scripts.data_convert.convert_common import unique
from scripts.config import QUESTION_FILE_JSON, TEXT_RAW_FIELD_NAME
from scripts.data_convert.convert_common import readQueries

QUERY_BATCH_SIZE=32

def jaccard(toks1, toks2):
    set1 = set(toks1)
    set2 = set(toks2)
    totQty = len(set1.union(set2))
    if totQty == 0:
        return 0
    return float(len(set1.intersection(set2))) / totQty

def getTokenIds(tokenizer, text):
    toks = tokenizer.tokenize(text)
    toks = [tokenizer.vocab[t] for t in toks]
    return toks

def toksToStr(arr):
    return ' '.join([str(k) for k in arr])

def strToNMSLIBVect(tokenizer, text):
    """Converts to a string that can be fed to NMSLIB"""
    lst = unique(getTokenIds(tokenizer, text))
    lst.sort()
    return toksToStr(lst)


def readSampleQueries(dataDir,
                      input_subdir1, sample_prob1,
                      input_subdir2, sample_prob2):
    qpath1 = os.path.join(dataDir, input_subdir1, QUESTION_FILE_JSON)
    fullQueryList1 = readQueries(qpath1)
    sampleQueryList1 = np.random.choice(fullQueryList1,
                                        size=int(len(fullQueryList1) * sample_prob1),
                                        replace=False)
    print('Read %d queries from %s sampled %d' %
          (len(fullQueryList1), qpath1, len(sampleQueryList1)))

    qpath2 = os.path.join(dataDir, input_subdir2, QUESTION_FILE_JSON)
    fullQueryList2 = readQueries(qpath2)
    sampleQueryList2 = np.random.choice(fullQueryList2,
                                        size=int(len(fullQueryList2) * sample_prob2),
                                        replace=False)
    print('Read %d queries from %s sampled %d' %
          (len(fullQueryList2), qpath2, len(sampleQueryList2)))

    return sampleQueryList1, sampleQueryList2


def createJaccardIndex(useHnsw, tokenizer, queryList):
    if useHnsw:
        methodName = 'hnsw'
        M = 30
        efC = 200

        indexTimeParams = {'M': M, 'efConstruction': efC, 'indexThreadQty': 0, 'post': 0}
        efS = 1000
        queryTimeParams = {'efSearch': efS}
    else:
        methodName = 'brute_force'
        indexTimeParams = {}
        queryTimeParams = {}

    print('k-NN search method', methodName)

    index = nmslib.init(method=methodName,
                        space='jaccard_sparse',
                        data_type=nmslib.DataType.OBJECT_AS_STRING)

    for start in tqdm(range(0, len(queryList), QUERY_BATCH_SIZE), desc='reading query set'):
        dbatch = []
        for e in queryList[start:start + QUERY_BATCH_SIZE]:
            dbatch.append(strToNMSLIBVect(tokenizer, e[TEXT_RAW_FIELD_NAME]))

        if dbatch:
            index.addDataPointBatch(dbatch)
            dbatch = []

    print('# of data points to index', len(index))

    # Create an index
    start = time.time()
    index.createIndex(indexTimeParams)
    end = time.time()
    print('Index-time parameters', indexTimeParams)
    print('Indexing time = %f' % (end - start))

    print('Setting query-time parameters', queryTimeParams)
    index.setQueryTimeParams(queryTimeParams)

    return index