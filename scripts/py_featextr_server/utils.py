import pandas as pd
import numpy as np


# Loads embeddings stored in Glove-vector text format
def load_embeddings(file_name, sep='\t'):
    dt_frame = pd.read_csv(file_name, sep=sep, header=None)
    words = dt_frame[0].values
    dt_frame.drop(0, axis=1, inplace=True)
    return words, dt_frame.values.astype(np.float32)


def create_embed_map(words):
    res = dict()
    for i in range(len(words)):
        res[words[i]] = i
    return res


# Unlike scipy cosine it doesn't choke on zero vectors
def robust_cosine_simil(x, y, eps=1e-10):
    sum_x = np.sqrt(np.sum(x * x))
    sum_y = np.sqrt(np.sum(y * y))
    sum_x = max(sum_x, eps)
    sum_y = max(sum_y, eps)
    return np.sum(x * y) / (sum_x * sum_y)
