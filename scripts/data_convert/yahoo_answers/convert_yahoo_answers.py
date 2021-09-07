#!/usr/bin/env python
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
import os
import json
import argparse

"""
    This script converts data in a quasi Yahoo Answers format (with preamble removed).
    It expects a collection produced by the Yahoo Answers collection splitter: split_yahoo_answers_input.sh
"""
from flexneuart.text_proc.formats import proc_yahoo_answers_record, SimpleXmlRecIterator
from flexneuart.text_proc.parse import SpacyTextParser, get_retokenized
from flexneuart.io import FileWrapper
from flexneuart.io.qrels import gen_qrel_str
from flexneuart.io.stopwords import read_stop_words

from flexneuart.data_convert import add_bert_tok_args, create_bert_tokenizer_if_needed, \
                            OUT_BITEXT_PATH_OPT, OUT_BITEXT_PATH_OPT_META, OUT_BITEXT_PATH_OPT_HELP
from flexneuart.config import SPACY_MODEL, ANSWER_FILE_JSON, BITEXT_QUESTION_PREFIX,\
                            QREL_FILE, BITEXT_ANSWER_PREFIX, REPORT_QTY
from flexneuart.config import DOCID_FIELD, QUESTION_FILE_JSON, TEXT_FIELD_NAME, \
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
add_bert_tok_args(parser)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inp_file_name = args.input

out_main_dir = args.out_main_path
out_bitext_dir = arg_vars[OUT_BITEXT_PATH_OPT]

bert_tokenizer = create_bert_tokenizer_if_needed(args)

fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TEXT_RAW_FIELD_NAME]
# It doesn't make sense to have bitext data for a raw-text field,
# because MGIZA needs data to be white-space tokenized,
# however, it makes sense to create a bitext set for a BERT-tokenized field.
bitext_fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME]

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
