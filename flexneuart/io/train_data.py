#
# This code is a modified version of CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
from tqdm import tqdm
from flexneuart.io import open_with_default_enc


def read_datafiles(file_names):
    """Read train and test files in CEDR format.

    :param file_names:   an array of file file name
    :return: a dataset, which is tuple of two dictionaries representing queries and documents, respectively.
    """
    queries = {}
    docs = {}
    for file_name in file_names:
        with open_with_default_enc(file_name, 'rt') as file:
            for ln, line in enumerate(tqdm(file, desc='loading datafile (by line)', leave=False)):
                line = line.rstrip()
                if not line:
                    tqdm.write(f'Skipping empty line: {ln+1}')
                    continue
                cols = line.split('\t')
                field_qty = len(cols)
                if field_qty != 3:
                    tqdm.write(f'skipping line {ln+1} because it has wrong # of fields: "{field_qty}"')
                    continue
                c_type, c_id, c_text = cols
                assert c_type in ('query', 'doc')
                if c_type == 'query':
                    queries[c_id] = c_text
                if c_type == 'doc':
                    docs[c_id] = c_text

    return queries, docs


def read_pairs_dict(file_name):
    """
        Read training pairs and scores provided by a candidate generator.
       This is almost a CEDR format except for the optional candidate generator scores.

    :param file_name: the name of the file
    :return:    Training pairs in the dictionary of dictionary formats.
                Candidate generator scores are  values of the inner-most dictionary.
                If the score isn't provide its value is set to zero.
    """
    result = {}
    with open_with_default_enc(file_name, 'rt') as file:
        for ln, line in enumerate(tqdm(file, desc='loading pairs (by line)', leave=False)):
            line = line.rstrip()
            if not line:
                tqdm.write(f'Skipping empty line: {ln+1}')
                continue
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
    with open_with_default_enc(file_name, 'w') as outf:
        for qid, docid_dict in train_pairs.items():
            for did, score in docid_dict.items():
                outf.write(f'{qid}\t{did}\t{score}\n')


def train_item_qty_upper_bound(train_pairs, epoch_repeat_qty):
    """
       This function estimates the number of training steps. If a query always
       has a positive example, this estimate should be accurate. Otherwise,
       it is only an upper bound.
       This function (together with our approach to iterate over training data)
       is quite hacky and possibly a better solution would be a possibility
       to define a number of training steps per epoch explicitly.
    """
    return epoch_repeat_qty * len(list(train_pairs.keys()))
