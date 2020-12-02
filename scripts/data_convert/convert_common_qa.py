import argparse
import sys
import os
import json
import argparse
import numpy as np

from scripts.config import SPACY_MODEL
from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import readStopWords, addRetokenizedField, \
                                                BERT_TOK_OPT, BERT_TOK_OPT_HELP
from scripts.config import DOCID_FIELD, QUESTION_FILE_JSON, BERT_BASE_MODEL, \
                            TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, \
                            TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, \
                            ANSWER_LIST_FIELD_NAME, STOPWORD_FILE
from pytorch_pretrained_bert import BertTokenizer


def outputQuestionList(nlp, bertTokenizer, qlist, q_idx, outDir):
    """Save questions with the specified indices.

    :param nlp:                 a text processing object.
    :param bertTokenizer:       a bert tokenizer (can be None).
    :param qlist:               a *FULL* list of questions.
    :param q_idx:               a list of question indices (other questions are ignored)
    :param outDir:              an output directory

    """
    if len(q_idx):
        with open(os.path.join(outDir, QUESTION_FILE_JSON), 'w') as outFile:
            for idx in q_idx:
                qid, question, answerList = qlist[idx]

                questionLemmas, questionUnlemm = nlp.procText(question)

                question = question.lower()  # after NLP

                answerListProc = set()

                for answ in answerList:
                    answLemmas, _ = nlp.procText(answ)
                    answerListProc.add(answLemmas)

                doc = {DOCID_FIELD: qid,
                       TEXT_FIELD_NAME: questionLemmas,
                       TEXT_UNLEMM_FIELD_NAME: questionUnlemm,
                       TEXT_RAW_FIELD_NAME: question,
                       ANSWER_LIST_FIELD_NAME: list(answerListProc)}
                addRetokenizedField(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bertTokenizer)
                docStr = json.dumps(doc) + '\n'
                outFile.write(docStr)


def convertAndSaveQueries(readingFunc, desc='convert questions'):
    """A high-level function function to parse input questions into queries.
       It reads command line arguments on its own, but delegates parsing of the input data
       to the function readingFunc.

    :param readingFunc: this function is expected to yield the following triples:
                        question ID, question text, and the list of answers (as unparsed text fragments)
    :param desc:        program desription
    """

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input', metavar='input file', help='input file',
                        type=str, required=True)
    parser.add_argument('--output_dir', metavar='out dir',
                        help='output directory',
                        type=str, required=True)
    parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)

    args = parser.parse_args()
    arg_vars = vars(args)
    print(args)

    bertTokenizer = None
    if BERT_TOK_OPT in arg_vars:
        print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
        bertTokenizer = BertTokenizer.from_pretrained(BERT_BASE_MODEL)

    stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
    print(stopWords)
    nlp = SpacyTextParser(SPACY_MODEL, stopWords, keepOnlyAlphaNum=True, lowerCase=True, enablePOS=False)

    qlist = []

    for questionId, questionText, answerList in readingFunc(args.input):
        qlist.append((questionId, questionText, answerList))

    qty = len(qlist)
    qids = np.arange(qty)

    outputQuestionList(nlp, bertTokenizer, qlist, qids, args.output_dir)


def convertAndSaveParagraphs(readingFunc):
    """A high-level function function to parse input paragraphs.
       It reads command line arguments on its own, but delegates parsing of the input data
       to the function readingFunc.

    :param readingFunc: this function is expected to yield a string per paragraph, which
                        has the following TAB-separated entities:
                        question ID, question text, and the list of answers (as unparsed text fragments)
    """

