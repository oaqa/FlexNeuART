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

"""
    This file defines a number of constants/settings.

    *IMPORTANT NOTEs*:

    1. Some constants in this config "mirror" respective shell constants.
    2. Mirroring constants should have the same values in both shell and Python.
    3. The same applies to a function that re-implements/mirrors a shell-script functionality.

    See the following files for the reference:

    1. config.sh
    2. common_proc.sh
"""

import sys
import os

# This must be a tuple of ints
MIN_PYTHON_VERSION=(3, 6)

SPACY_MODEL = 'en_core_web_sm'

# Constants defined here mostly can't be modified as they should be in sync with Java and bash code
STOPWORD_FILE = 'data/stopwords.txt'
MAX_DOC_SIZE=32768 # 32 K of text is enough for most applications
MAX_PASS_SIZE=8192 # Let's have passages fitting into a single BERT input chunk
REPORT_QTY=10000

# A size of the chunk passed used in imap
# it should be sufficiently large, but not too large
IMAP_PROC_CHUNK_QTY=500

DEVICE_CPU = 'cpu'
DEFAULT_DEVICE_GPU = 'cuda'

BERT_BASE_MODEL='bert-base-uncased'
MSMARCO_MINILM_L2='cross-encoder/ms-marco-MiniLM-L-2-v2'

QREL_FILE = 'qrels.txt'
QUESTION_FILE_PREFIX='QuestionFields'
QUESTION_FILE_JSON = f'{QUESTION_FILE_PREFIX}.jsonl'
QUESTION_FILE_BIN = f'{QUESTION_FILE_PREFIX}.bin'
ANSWER_FILE_PREFIX='AnswerFields'
ANSWER_FILE_JSON = f'{ANSWER_FILE_PREFIX}.jsonl'
ANSWER_FILE_BIN = f'{ANSWER_FILE_PREFIX}.bin'
ANSWER_FILE_JSONL_GZ = f'{ANSWER_FILE_JSON}.gz' # We'd like to keep it compressed

DOCID_FIELD = 'DOCNO'

TEXT_FIELD_NAME = 'text'
TEXT_STEMMED_FIELD_NAME = 'text_stems'
TEXT_UNLEMM_FIELD_NAME = 'text_unlemm'
TEXT_BERT_TOKENIZED_NAME = 'text_bert_tok'
TITLE_FIELD_NAME = 'title'
TITLE_UNLEMM_FIELD_NAME = 'title_unlemm'
TITLE_FIELD_NAME = 'title'
TEXT_RAW_FIELD_NAME = 'text_raw'

ANSWER_LIST_FIELD_NAME = 'answer_list'

# We don't really use it yet
DEFAULT_ENCODING = 'utf-8'

# bitext naming conventions
BITEXT_QUESTION_PREFIX = 'question_'
BITEXT_ANSWER_PREFIX = 'answer_'

MAX_RELEV_GRADE=4

# Default number of iterations in GIZA EM algorithm
DEFAULT_GIZA_ITER_QTY=5

DEFAULT_TRAIN_SUBDIR='train_fusion'
DEFAULT_BITEXT_SUBDIR='bitext'

# Let's use STDOUT, it makes it easier to sync output
TQDM_FILE=sys.stdout
DEFAULT_MAX_QUERY_LEN=32
DEFAULT_MAX_DOC_LEN=512 - DEFAULT_MAX_QUERY_LEN - 4

DEFAULT_VAL_BATCH_SIZE=32

MAX_INT32=2147483647

class InputDataInfo:
    def __init__(self, index_subdirs, query_subdirs, data_file_name):
        self.index_subdirs = index_subdirs
        self.query_subdirs = query_subdirs
        self.data_file_name = data_file_name

def get_index_query_data_dirs(top_dir : str) -> InputDataInfo:
    """
       This is a 'discovery' function, which (1) finds all data and query files
       in a given top-level directory, (2) checks basic consistency. For example,
       if some query/question files have binary data data and some do not, it
       will raise an exception.

       Note that the files are supposed to be found in immediate sub-directories,
       i.e., there will be no recursive search.

        :param top_dir: a top-level directory whose sub-directories contain data
                       and/or query-files.
        :return collection information object
    """
    subdir_list = [subdir for subdir in os.listdir(top_dir) if os.path.isdir(subdir)]

    index_subdirs = []
    query_subdirs = []
    data_file_name = None

    for subdir in subdir_list:
        print(f'Checking sub-directory: {subdir}')

        has_data = False
        for suff in ["", ".gz", ".bz2"]:
            data_fn = ANSWER_FILE_JSON + suff
            fn = os.path.join(top_dir, subdir, data_fn)
            if os.path.exists(fn) and os.path.isfile(fn):
                print(f'Found indexable data file: {fn}')
                if data_file_name is None:
                    data_file_name = data_fn
                elif data_file_name != data_fn:
                    raise Exception(f'Inconsistent compression of data files: {data_file_name} and {data_fn}')
        if has_data:
            index_subdirs.append(subdir)
        else:
            fn_bin = os.path.join(subdir, ANSWER_FILE_BIN)
            if os.path.exists(fn_bin):
                raise Exception(f'Inconsistent data setup in sub-directory {subdir}: ' +
                                f'the binary data file is present, but no (un)compressed JSONL file')

        fn = os.path.join(top_dir, subdir, QUESTION_FILE_JSON)
        fn_bin = os.path.join(top_dir, subdir, QUESTION_FILE_BIN)
        if os.path.exists(fn):
            query_subdirs.append(subdir)
        else:
            if os.path.exists(fn_bin):
                raise Exception(f'Inconsistent query setup in sub-directory {subdir}:' +
                                 'the binary query file is present, but no (un)compressed JSONL file')

    return InputDataInfo(index_subdirs=index_subdirs, query_subdirs=query_subdirs, data_file_name=data_file_name)

