import pandas as pd
import numpy as np


# Loads embeddings stored in Glove-vector text format
def loadEmbeddings(fileName, sep='\t'):
    dtFrame = pd.read_csv(fileName, sep=sep, header=None)
    words = dtFrame[0].values
    dtFrame.drop(0, axis=1, inplace=True)
    return words, dtFrame.values.astype(np.float32)


def createEmbedMap(words):
    res = dict()
    for i in range(len(words)):
        res[words[i]] = i
    return res


# Unlike scipy cosine it doesn't choke on zero vectors
def robustCosineSimil(x, y, eps=1e-10):
    sumX = np.sqrt(np.sum(x * x))
    sumY = np.sqrt(np.sum(y * y))
    sumX = max(sumX, eps)
    sumY = max(sumY, eps)
    return np.sum(x * y) / (sumX * sumY)
