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

    *IMPORTANT NOTE*: When a constant is used in the shell script, it must have have the same value.
    See the following files for the reference:

    1. config.sh
    2. common_proc.sh
"""

import sys

SPACY_MODEL = 'en_core_web_sm'
PYTORCH_DISTR_BACKEND='gloo'

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
QUESTION_FILE_JSON = 'QuestionFields.jsonl'
ANSWER_FILE_JSON = 'AnswerFields.jsonl.gz' # We'd like to keep it compressed

DOCID_FIELD = 'DOCNO'

TEXT_FIELD_NAME = 'text'
TEXT_UNLEMM_FIELD_NAME = 'text_unlemm'
TEXT_BERT_TOKENIZED_NAME = 'text_bert_tok'
TITLE_FIELD_NAME = 'title'
TITLE_UNLEMM_FIELD_NAME = 'title_unlemm'
TITLE_FIELD_NAME = 'title'
TEXT_RAW_FIELD_NAME = 'text_raw'

ANSWER_LIST_FIELD_NAME = 'answer_list'

# We don't really use it
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