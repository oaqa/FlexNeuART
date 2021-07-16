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
# Constants, which *SHOULD* also be in sync with config.sh and common_proc.sh
#
SPACY_MODEL = 'en_core_web_sm'

# Constants defined here mostly can't be modified as they should be in sync with Java and bash code
STOPWORD_FILE = 'data/stopwords.txt'
MAX_DOC_SIZE=32768 # 32 K of text is enough for most applications
REPORT_QTY=10000

# A size of the chunk passed used in imap
# it should be sufficiently large, but not too large
IMAP_PROC_CHUNK_QTY=500

DEVICE_CPU = 'cpu'

BERT_BASE_MODEL='bert-base-uncased'
BERT_LARGE_MODEL='bert-large-uncased'

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

DEFAULT_ENCODING = 'utf-8'

# bitext naming conventions
BITEXT_QUESTION_PREFIX = 'question_'
BITEXT_ANSWER_PREFIX = 'answer_'

MAX_RELEV_GRADE=4

# Default number of iterations in GIZA EM algorithm
DEFAULT_GIZA_ITER_QTY=5

DEFAULT_TRAIN_SUBDIR='train_fusion'
DEFAULT_BITEXT_SUBDIR='bitext'
