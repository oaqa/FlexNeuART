#!/usr/bin/env python3
import sys, pickle
import argparse
import matchzoo as mz
from matchzoo.data_pack import pack
import numpy as np
import pandas as pd

sys.path.append('scripts/py_server')
sys.path.append('scripts/data')
sys.path.append('.')

from base_server import *
from matchzoo_reader import *


class MatchZooQueryHandler(BaseQueryHandler):
  # Exclusive==True means that only one getScores
  # function is executed at at time
  def __init__(self, modelDir, dtProcDir, debuPrint):
    super().__init__(exclusive=True)

    with open(dtProcDir, 'rb') as f:
      self.prep = pickle.load(f)

    self.model = mz.load_model(modelDir)
    self.model.backend.summary()
    self.debugPrint = debugPrint

  # This function needs to be overriden
  def computeScoresOverride(self, query, docs):
    queryText = self.concatTextEntryWords(query)
    if self.debugPrint:
      print('getScores', queryText)

    queryIdArr = []
    queryTextArr = []
    docTextArr = []
    docIdArr = []
    labelArr = []

    for e in docs:
      docIdArr.append(e.id)
      docTextArr.append(self.concatTextEntryWords(e))
      queryTextArr.append(queryText)
      queryIdArr.append('fake_qid')
      labelArr.append(0)


    dataRaw = pd.DataFrame({'id_left' : queryIdArr,
                         'text_left' : queryTextArr,
                         'id_right' : docIdArr,
                         'text_right' : docTextArr,
                         'label' : labelArr})

    dataTestPacked = pack(dataRaw)

    dataTestProc = self.prep.transform(dataTestPacked)

    dataForModel, _ = dataTestProc.unpack()

    preds = self.model.predict(dataForModel)

    sampleRet = {}
    for k in range(len(docs)):
      e = docs[k]
      score = preds[k]
      if self.debugPrint:
        print(score, self.textEntryToStr(e))
      sampleRet[e.id] = score

    return sampleRet

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Serving MatchZoo models.')

  parser.add_argument('--model', metavar='MatchZoo model',
                      required=True, type=str,
                      help='Model directory')

  parser.add_argument('--debug_print', actiona='store_true',
                      help='Provide debug output')

  parser.add_argument('--dtproc_model', metavar='data processing model',
                      required=True, type=str,
                      help='Pickled data processor file')

  parser.add_argument('--port', metavar='server port',
                      required=True, type=int,
                      help='Server port')

  parser.add_argument('--host', metavar='server host',
                      default='127.0.0.1', type=str,
                      help='server host addr to bind the port')

  args = parser.parse_args()


  multiThreaded=False #
  startQueryServer(args.host, args.port, multiThreaded, MatchZooQueryHandler(modelDir=args.model,
                                                                             dtProcDir=args.dtproc_model,
                                                                             debugPrint=args.debug_print))

