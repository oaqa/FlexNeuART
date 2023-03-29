#
# This code is a modified version of CEDR: https://github.com/Georgetown-IR-Lab/cedr
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#
from tqdm import tqdm
from flexneuart.io import open_with_default_enc, FileWrapper
from flexneuart.io.qrels import QrelEntry, qrel_entry2_str

# This things are hard-coded and must match Java and shell scripts
DATA_QUERY = 'data_query.tsv'
DATA_DOCS = 'data_docs.tsv'
TRAIN_PAIRS = 'train_pairs.tsv'
TEST_RUN = 'test_run.txt'
QRELS = 'qrels.txt'

DATA_TYPE_QUERY = 'query'
DATA_TYPE_DOC = 'doc'

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
                assert c_type in (DATA_TYPE_QUERY, DATA_TYPE_DOC)
                if c_type == DATA_TYPE_QUERY:
                    queries[c_id] = c_text
                if c_type == DATA_TYPE_DOC:
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

def write_pairs_dict_open_file(train_pairs, outf):
    """Write training pairs: out_f must be an open file.

    :param train_pairs:   training data dictionary of dictionaries.
    :param outf          an open file.
    """
    for qid, docid_dict in train_pairs.items():
        for did, score in docid_dict.items():
            outf.write(f'{qid}\t{did}\t{score}\n')


def write_pairs_dict(train_pairs, file_name):
    """Write training pairs.

    :param train_pairs:   training data dictionary of dictionaries.
    :param file_name:     output file name
    """
    with FileWrapper(file_name, 'w') as outf:
        write_pairs_dict_open_file(train_pairs, outf)


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


def write_filtered_datafiles(out_f, data, data_type, id_filter_set):
    # File must be opened
    print(f'Writing to {out_f.name} type: {data_type}')
    qty = 0
    for id, v in data.items():
        if id in id_filter_set:
            out_f.write(f'{data_type}\t{id}\t{v}\n')
            qty += 1

    print(f'{qty} items written')


def write_filtered_train_pairs(out_fn, train_pairs_full, qid_filter_set):
    # File must be opened
    print(f'Writing train pairs to {out_fn.name}')
    qty = 0
    train_pairs_filtered = {}
    for qid, did_dict in train_pairs_full.items():
        if qid in qid_filter_set:
            train_pairs_filtered[qid] = did_dict
            qty += len(did_dict)

    write_pairs_dict_open_file(train_pairs_filtered, out_fn)

    print(f'# of queris in a full set: {len(train_pairs_full)} filtered set: {len(train_pairs_filtered)}')
    print(f'{qty} items written')


def write_filtered_qrels(out_f, qrels, qid_filter_set):
    print(f'Writing qrels to {out_f.name}')
    # File must be opened
    qty = 0
    for qid, did_rel_dict in qrels.items():
        if qid in qid_filter_set:
            for did, grade in did_rel_dict.items():
                e = QrelEntry(query_id=qid, doc_id=did, rel_grade=grade)
                out_f.write(qrel_entry2_str(e) + '\n')
                qty += 1

    print(f'{qty} items written')

