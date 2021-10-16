#
# This code is a modified version of CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
from tqdm import tqdm


def read_datafiles(files):
    """Read train and test files in CEDR format.

    :param files:   an array of file objects, which represent queries or documents (in any order)
    :return: a dataset, which is tuple of two dictionaries representing queries and documents, respectively.
    """
    queries = {}
    docs = {}
    for file in files:
        for line in tqdm(file, desc='loading datafile (by line)', leave=False):
            cols = line.rstrip().split('\t')
            if len(cols) != 3:
                tqdm.write(f'skipping line: `{line.rstrip()}`')
                continue
            c_type, c_id, c_text = cols
            assert c_type in ('query', 'doc')
            if c_type == 'query':
                queries[c_id] = c_text
            if c_type == 'doc':
                docs[c_id] = c_text
    return queries, docs


def read_pairs_dict(file):
    """
        Read training pairs and scores provided by a candidate generator.
       This is almost a CEDR format except for the optional candidate generator scores.

    :param file: an open file, not a file name!
    :return:    Training pairs in the dictionary of dictionary formats.
                Candidate generator scores are  values of the inner-most dictionary.
                If the score isn't provide its value is set to zero.
    """
    result = {}
    for ln, line in enumerate(tqdm(file, desc='loading pairs (by line)', leave=False)):
        fields = line.split()
        if not len(fields) in [2, 3]:
            raise Exception(f'Wrong # of fields {len(fields)} in file {file}, line #: {ln+1}')
        qid, docid = fields[0: 2]
        if len(fields) == 3:
            score = fields[2]
        else:
            score = 0

        result.setdefault(qid, {})[docid] = float(score)

    return result


def write_pairs_dict(train_pairs, file_name):
    """Write training pairs.

    :param train_pairs:   training data dictionary of dictionaries.
    :param file_name:     output file name
    """
    with open(file_name, 'w') as outf:
        for qid, docid_dict in train_pairs.items():
            for did, score in docid_dict.items():
                outf.write(f'{qid}\t{did}\t{score}\n')


def train_item_qty_upper_bound(train_pairs):
    return len(list(train_pairs.keys()))
