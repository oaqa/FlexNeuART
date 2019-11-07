#!/usr/bin/env python
import sys, os
import json
import argparse
import numpy as np

sys.path.append('scripts')
from data_convert.text_proc import *
from data_convert.convert_common import *

parser = argparse.ArgumentParser(description='Convert a collection of SQuAD questions.')
parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--split_prob', metavar='1st split prob', 
                    help='First split probability, if it is equal to one, there is not output for the second split', type=float, default=1.0)
parser.add_argument('--output_dir_split1', metavar='1st split out dir', 
                    help='Output directory for the first split',
                    type=str, required=True)
parser.add_argument('--output_dir_split2', metavar='2d split out dir', 
                    help='Output directory for the second split',
                    type=str, default=None)

args = parser.parse_args()
print(args)

fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TEXT_RAW_FIELD_NAME]

stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)
nlp = SpacyTextParser(SPACY_MODEL, stopWords, keepOnlyAlphaNum=True, lowerCase=True, enablePOS=False)

root = json.load(open(args.input))

qlist = []

for page in root["data"]:
  for para in page["paragraphs"]:
    for qinfo in para["qas"]:
      answerList =[e["text"] for e in qinfo["answers"]]
      qlist.append( (qinfo["id"], qinfo["question"], answerList) )

qty = len(qlist)

qids = np.arange(qty)
np.random.seed(0)
np.random.shuffle(qids)
if args.output_dir_split2 is not None:
  qty1 = int(qty * args.split_prob)
else:
  print('Do not create a random sample, b/c the second sub-directory is not specified!')
  qty1 = qty

def doOutput(nlp, qlist, qids, outPref):
  if len(qids):
    with open(os.path.join(outPref, QUESTION_FILE_JSON), 'w') as outFile:
      for i in qids:
        qid, question, answerList = qlist[i]

        questionLemmas, questionUnlemm = nlp.procText(question)

        question = question.lower() # after NLP

        answerListProc = set()
  
        for answ in answerList:
          answLemmas, _ = nlp.procText(answ)
          answerListProc.add(answLemmas)

        doc = {DOCID_FIELD : qid,
              TEXT_FIELD_NAME : questionLemmas,
              TEXT_UNLEMM_FIELD_NAME : questionUnlemm,
              TEXT_RAW_FIELD_NAME : question,
              ANSWER_LIST_FIELD_NAME : list(answerListProc)}
        docStr = json.dumps(doc) + '\n'
        outFile.write(docStr)

  

doOutput(nlp, qlist, qids[0:qty1], args.output_dir_split1)
if args.output_dir_split2 is not None:
  doOutput(nlp, qlist, qids[qty1:qty], args.output_dir_split2)

