#!/usr/bin/env python
import sys, os
import json
import argparse
import pytorch_pretrained_bert

# This collection converts data in a quasi Yahoo Answers format (with preamble removed).
# It expects a collection either produced by other programs such as
# 1) a Stack Overflow converter script: scripts/data_convert/stack_overflow/convert_stackoverflow_full.sh
# 2) Yahoo Answers collection splitter:

sys.path.append('.')


from scripts.data_convert.text_proc import SpacyTextParser
from scripts.common_eval import gen_qrel_str
from scripts.data_convert.convert_common import FileWrapper, read_stop_words, \
                                                BERT_TOK_OPT, BERT_TOK_OPT_HELP, \
                                                OUT_BITEXT_PATH_OPT, OUT_BITEXT_PATH_OPT_META, OUT_BITEXT_PATH_OPT_HELP, \
                                                get_retokenized, SimpleXmlRecIterator,\
                                                proc_yahoo_answers_record
from scripts.config import SPACY_MODEL, BERT_BASE_MODEL, ANSWER_FILE_JSON, BITEXT_QUESTION_PREFIX,\
                            QREL_FILE, BITEXT_ANSWER_PREFIX, REPORT_QTY
from scripts.config import DOCID_FIELD, QUESTION_FILE_JSON, TEXT_FIELD_NAME, \
                            TEXT_UNLEMM_FIELD_NAME, TEXT_RAW_FIELD_NAME, \
                            STOPWORD_FILE, TEXT_BERT_TOKENIZED_NAME, MAX_RELEV_GRADE

parser = argparse.ArgumentParser(description='Convert a previously split collection in Yahoo Answers format.')
parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--out_main_path', metavar='main output directory',
                    help='main output directory, which includes indexable data and QRELs',
                    type=str, required=True)
parser.add_argument('--' + OUT_BITEXT_PATH_OPT, metavar=OUT_BITEXT_PATH_OPT_META,
                    help=OUT_BITEXT_PATH_OPT_HELP,
                    type=str, default=None)
parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inp_file_name = args.input

out_main_dir = args.out_main_path
out_bitext_dir = arg_vars[OUT_BITEXT_PATH_OPT]

bert_tokenizer = None

fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TEXT_RAW_FIELD_NAME]
# It doesn't make sense to have bitext data for a raw-text field,
# because MGIZA needs data to be white-space tokenized,
# however, it makes sense to create a bitext set for a BERT-tokenized field.
bitext_fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME]

if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bert_tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)
    bitext_fields.append(TEXT_BERT_TOKENIZED_NAME)

if not os.path.exists(out_main_dir):
    os.makedirs(out_main_dir)

bi_quest_files = {}
bi_answ_files = {}

stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
print(stop_words)
nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True, enable_pos=False)

data_quest_file = open(os.path.join(out_main_dir, QUESTION_FILE_JSON), 'w')
# File wrapper can handle output gz files
data_answ_file = FileWrapper(os.path.join(out_main_dir, ANSWER_FILE_JSON), flags='w')
qrel_file = open(os.path.join(out_main_dir, QREL_FILE), 'w')

if out_bitext_dir:
    if not os.path.exists(out_bitext_dir):
        os.makedirs(out_bitext_dir)

    for fn in bitext_fields:
        bi_quest_files[fn] = open(os.path.join(out_bitext_dir, BITEXT_QUESTION_PREFIX + fn), 'w')
        bi_answ_files[fn] = open(os.path.join(out_bitext_dir, BITEXT_ANSWER_PREFIX + fn), 'w')

ln = 0
for rec_str in SimpleXmlRecIterator(inp_file_name, 'document'):
    ln += 1
    try:
        rec = proc_yahoo_answers_record(rec_str)
        if len(rec.answer_list) == 0:  # Ignore questions without answers
            continue

        question_orig = (rec.subject + ' ' + rec.content).strip()
        question_lc = question_orig.lower()
        qid = rec.uri

        question_lemmas, question_unlemm = nlp.proc_text(question_orig)


        question_bert_tok = None
        if bert_tokenizer:
            question_bert_tok = get_retokenized(bert_tokenizer, question_lc)

        doc = {DOCID_FIELD: qid,
               TEXT_FIELD_NAME: question_lemmas,
               TEXT_UNLEMM_FIELD_NAME: question_unlemm,
               TEXT_RAW_FIELD_NAME: question_orig}

        if question_bert_tok is not None:
            doc[TEXT_BERT_TOKENIZED_NAME] = question_bert_tok
        doc_str = json.dumps(doc) + '\n'
        data_quest_file.write(doc_str)

        for i in range(len(rec.answer_list)):
            aid = qid + '-' + str(i)
            answ_orig = rec.answer_list[i]
            answ_lc = answ_orig.lower()

            answ_lemmas, answ_unlemm = nlp.proc_text(answ_orig)

            # Doing it after lower-casing
            answ_bert_tok = None
            if bert_tokenizer:
                answ_bert_tok = get_retokenized(bert_tokenizer, answ_lc)

            doc = {DOCID_FIELD: aid,
                   TEXT_FIELD_NAME: answ_lemmas,
                   TEXT_UNLEMM_FIELD_NAME: answ_unlemm,
                   TEXT_RAW_FIELD_NAME: answ_orig}

            if answ_bert_tok is not None:
                doc[TEXT_BERT_TOKENIZED_NAME] = answ_bert_tok

            doc_str = json.dumps(doc) + '\n'
            data_answ_file.write(doc_str)

            rel_grade = MAX_RELEV_GRADE - int(i != rec.best_answer_id)
            qrel_file.write(gen_qrel_str(qid, aid, rel_grade) + '\n')

            if bi_quest_files and bi_answ_files:
                bi_quest_files[TEXT_FIELD_NAME].write(question_lemmas + '\n')
                bi_quest_files[TEXT_UNLEMM_FIELD_NAME].write(question_lemmas + '\n')

                bi_answ_files[TEXT_FIELD_NAME].write(answ_lemmas + '\n')
                bi_answ_files[TEXT_UNLEMM_FIELD_NAME].write(answ_lemmas + '\n')

                if bert_tokenizer is not None:
                    bi_quest_files[TEXT_BERT_TOKENIZED_NAME].write(question_bert_tok + '\n')
                    bi_answ_files[TEXT_BERT_TOKENIZED_NAME].write(answ_bert_tok + '\n')

        if ln % REPORT_QTY == 0:
            print('Processed %d questions' % ln)

    except Exception as e:
        print(f'Error parsing record #{ln}, error msg: ' + str(e))

data_quest_file.close()
data_answ_file.close()
qrel_file.close()

for _, f in bi_quest_files.items():
    f.close()
for _, f in bi_answ_files.items():
    f.close()
