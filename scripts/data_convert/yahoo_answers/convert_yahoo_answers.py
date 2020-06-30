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
from scripts.common_eval import genQrelStr
from scripts.data_convert.convert_common import FileWrapper, readStopWords, \
                                                BERT_TOK_OPT, BERT_TOK_OPT_HELP, \
                                                getRetokenized, SimpleXmlRecIterator,\
                                                procYahooAnswersRecord
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
parser.add_argument('--out_bitext_path', metavar='optional bitext output directory',
                    help='An optional output directory to store bitext',
                    type=str, default='')
parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inpFileName = args.input

outMainDir = args.out_main_path
outBitextDir = args.out_bitext_path

bertTokenizer = None

fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TEXT_RAW_FIELD_NAME]
# It doesn't make sense to have bitext data for a raw-text field,
# because MGIZA needs data to be white-space tokenized,
# however, it makes sense to create a bitext set for a BERT-tokenized field.
bitext_fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME]

if BERT_TOK_OPT in arg_vars:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bertTokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)
    bitext_fields.append(TEXT_BERT_TOKENIZED_NAME)

if not os.path.exists(outMainDir):
    os.makedirs(outMainDir)

biQuestFiles = {}
biAnswFiles = {}

stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)
nlp = SpacyTextParser(SPACY_MODEL, stopWords, keepOnlyAlphaNum=True, lowerCase=True, enablePOS=False)

dataQuestFile = open(os.path.join(outMainDir, QUESTION_FILE_JSON), 'w')
# File wrapper can handle output gz files
dataAnswFile = FileWrapper(os.path.join(outMainDir, ANSWER_FILE_JSON), flags='w')
qrelFile = open(os.path.join(outMainDir, QREL_FILE), 'w')

if outBitextDir:
    if not os.path.exists(outBitextDir):
        os.makedirs(outBitextDir)

    for fn in bitext_fields:
        biQuestFiles[fn] = open(os.path.join(outBitextDir, BITEXT_QUESTION_PREFIX + fn), 'w')
        biAnswFiles[fn] = open(os.path.join(outBitextDir, BITEXT_ANSWER_PREFIX + fn), 'w')

ln = 0
for recStr in SimpleXmlRecIterator(inpFileName, 'document'):
    ln += 1
    try:
        rec = procYahooAnswersRecord(recStr)
        if len(rec.answerList) == 0:  # Ignore questions without answers
            continue

        question = (rec.subject + ' ' + rec.content).strip()
        qid = rec.uri

        question_lemmas, question_unlemm = nlp.procText(question)

        question = question.lower()  # after NLP

        question_bert_tok = None
        if bertTokenizer:
            question_bert_tok = getRetokenized(bertTokenizer, question)

        doc = {DOCID_FIELD: qid,
               TEXT_FIELD_NAME: question_lemmas,
               TEXT_UNLEMM_FIELD_NAME: question_unlemm,
               TEXT_RAW_FIELD_NAME: question}
        if question_bert_tok is not None:
            doc[TEXT_BERT_TOKENIZED_NAME] = question_bert_tok
        docStr = json.dumps(doc) + '\n'
        dataQuestFile.write(docStr)

        for i in range(len(rec.answerList)):
            aid = qid + '-' + str(i)
            answ = rec.answerList[i]
            answ_lemmas, answ_unlemm = nlp.procText(answ)

            answ = answ.lower()  # after NLP

            # Doing it after lower-casing
            answ_bert_tok = None
            if bertTokenizer:
                answ_bert_tok = getRetokenized(bertTokenizer, answ)

            doc = {DOCID_FIELD: aid,
                   TEXT_FIELD_NAME: answ_lemmas,
                   TEXT_UNLEMM_FIELD_NAME: answ_unlemm,
                   TEXT_RAW_FIELD_NAME: answ}

            if answ_bert_tok is not None:
                doc[TEXT_BERT_TOKENIZED_NAME] = answ_bert_tok

            docStr = json.dumps(doc) + '\n'
            dataAnswFile.write(docStr)

            relGrade = MAX_RELEV_GRADE - int(i != rec.bestAnswerId)
            qrelFile.write(genQrelStr(qid, aid, relGrade) + '\n')

            if biQuestFiles and biAnswFiles:
                biQuestFiles[TEXT_FIELD_NAME].write(question_lemmas + '\n')
                biQuestFiles[TEXT_UNLEMM_FIELD_NAME].write(question_lemmas + '\n')

                biAnswFiles[TEXT_FIELD_NAME].write(answ_lemmas + '\n')
                biAnswFiles[TEXT_UNLEMM_FIELD_NAME].write(answ_lemmas + '\n')

                if bertTokenizer is not None:
                    biQuestFiles[TEXT_BERT_TOKENIZED_NAME].write(question_bert_tok + '\n')
                    biAnswFiles[TEXT_BERT_TOKENIZED_NAME].write(answ_bert_tok + '\n')

        if ln % REPORT_QTY == 0:
            print('Processed %d questions' % ln)

    except Exception as e:
        print(f'Error parsing record #{ln}, error msg: ' + str(e))

dataQuestFile.close()
dataAnswFile.close()
qrelFile.close()

for _, f in biQuestFiles.items():
    f.close()
for _, f in biAnswFiles.items():
    f.close()
