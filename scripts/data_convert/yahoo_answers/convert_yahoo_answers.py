#!/usr/bin/env python
import sys, os
import json
import argparse

sys.path.append('scripts')
from data_convert.text_proc import *
from data_convert.convert_common import *

parser = argparse.ArgumentParser(description='Convert a collection in Yahoo Answers format.')
parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--out_main_path', metavar='main output directory',
                    help='main output directory, which includes indexable data and QRELs',
                    type=str, required=True)
parser.add_argument('--out_bitext_path', metavar='optional bitext output directory',
                    help='An optional output directory to store bitext',
                    type=str, default='')


args = parser.parse_args()
print(args)

inpFileName = args.input

outMainDir = args.out_main_path
outBitextDir = args.out_bitext_path

if not os.path.exists(outMainDir):
  os.makedirs(outMainDir)

fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TEXT_RAW_FIELD_NAME]

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

  for fn in fields:
    biQuestFiles[fn]=open(os.path.join(outBitextDir, BITEXT_QUESTION_PREFIX + fn), 'w')
    biAnswFiles[fn]=open(os.path.join(outBitextDir, BITEXT_ANSWER_PREFIX + fn), 'w')

ln = 0
for recStr in SimpleXmlRecIterator(inpFileName, 'document'):
  ln += 1
  try:
    rec = procYahooAnswersRecord(recStr)
    if len(rec.answerList) == 0: # Ignore questions without answers
      continue

    question = (rec.subject + ' ' + rec.content).strip()
    qid = rec.uri

    question_lemmas, question_unlemm = nlp.procText(question)

    question = question.lower() # after NLP

    doc = {DOCID_FIELD : qid,
          TEXT_FIELD_NAME : question_lemmas,
          TEXT_UNLEMM_FIELD_NAME : question_unlemm,
          TEXT_RAW_FIELD_NAME : question}
    docStr = json.dumps(doc) + '\n'
    dataQuestFile.write(docStr)

    for i in range(len(rec.answerList)):
      aid = qid + '-' + str(i)
      answ = rec.answerList[i]
      answ_lemmas, answ_unlemm = nlp.procText(answ)

      answ = answ.lower() # after NLP

      doc = {DOCID_FIELD : aid,
             TEXT_FIELD_NAME: answ_lemmas,
             TEXT_UNLEMM_FIELD_NAME: answ_unlemm,
             TEXT_RAW_FIELD_NAME: answ}
      docStr = json.dumps(doc) + '\n'
      dataAnswFile.write(docStr)

      relGrade = MAX_RELEV_GRADE - int(i != rec.bestAnswerId)
      qrelFile.write(genQrelStr(qid, aid, relGrade) + '\n')

      if biQuestFiles and biAnswFiles:
        biQuestFiles[TEXT_FIELD_NAME].write(question_lemmas + '\n')
        biQuestFiles[TEXT_UNLEMM_FIELD_NAME].write(question_lemmas + '\n')
        biQuestFiles[TEXT_RAW_FIELD_NAME].write(question + '\n')

        biAnswFiles[TEXT_FIELD_NAME].write(answ_lemmas + '\n')
        biAnswFiles[TEXT_UNLEMM_FIELD_NAME].write(answ_lemmas + '\n')
        biAnswFiles[TEXT_RAW_FIELD_NAME].write(answ + '\n')


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
