# Constants, which should also be in sinc with config.sh and common_proc.sh

SPACY_MODEL = 'en_core_web_sm'

# Constants defined here mostly can't be modified as they should be in sync with Java and bash code
STOPWORD_FILE = 'data/stopwords.txt'
MAX_DOC_SIZE=16536 # 16 K should be more than enough!
REPORT_QTY=10000

# A size of the chunk passed used in imap
# it should be sufficiently large, but not too large
IMAP_PROC_CHUNK_QTY=256

BERT_BASE_MODEL='bert-base-uncased'

QREL_FILE = 'qrels.txt'
QUESTION_FILE_JSON = 'QuestionFields.jsonl'
ANSWER_FILE_JSON = 'AnswerFields.jsonl.gz' # We'd like to keep it compressed

DOCID_FIELD = 'DOCNO'

TEXT_FIELD_NAME = 'text'
TEXT_UNLEMM_FIELD_NAME = 'text_unlemm'
TEXT_BERT_TOKENIZED_NAME = 'text_bert_tok'
TITLE_FIELD_NAME = 'title'
TITLE_UNLEMM_FIELD_NAME = 'title_unlemm'
TEXT_RAW_FIELD_NAME = 'text_raw'

ANSWER_LIST_FIELD_NAME = 'answer_list'

DEFAULT_ENCODING = 'utf-8'

# bitext naming conventions
BITEXT_QUESTION_PREFIX = 'question_'
BITEXT_ANSWER_PREFIX = 'answer_'

MAX_RELEV_GRADE=4
