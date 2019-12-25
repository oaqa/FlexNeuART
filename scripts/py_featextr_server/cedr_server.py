#!/usr/bin/env python
import sys, pickle
import argparse
import numpy as np

sys.path.append('scripts/py_featextr_server')
sys.path.append('scripts/data')
sys.path.append('scripts/cedr')

from base_server import *

# CEDR imports
import train
import data

# Let's use a small number so that we can run two servers on a single GPU
DEFAULT_BATCH_SIZE=8 

class CedrQueryHandler(BaseQueryHandler):
  # Exclusive==True means that only one getScores
  # function is executed at at time
  def __init__(self, modelType, modelWeights=None, batchSize=DEFAULT_BATCH_SIZE, debugPrint=False):
    super().__init__(exclusive=True)

    self.debugPrint = debugPrint
    self.batchSize = batchSize
    self.model = train.MODEL_MAP[modelType]().cuda()
    if modelWeights is not None:
      if self.debugPrint:
        print(f'Loading model {modelType} from {modelWeights}')    
      self.model.load(modelWeights)

    # need to be in the eval mode
    self.model.eval()

  # This function needs to be overridden
  def computeScoresFromRawOverride(self, query, docs):
    if self.debugPrint:
      print('getScores', query.id, query.text)

    queryData = { query.id : query.text }
    # Run maps queries to arrays of document IDs see iter_valid_records (train.py)
    run = { query.id : [e.id for e in docs] }

    docData = {}
    for e in docs:
      docData[e.id] = e.text

    sampleRet = {}

    if docData:

      # based on the code from run_model function (train.py)
      dataSet = queryData, docData 
      for records in data.iter_valid_records(self.model, dataSet, run, self.batchSize):
        scores = self.model(records['query_tok'],
                            records['query_mask'],
                            records['doc_tok'],
                            records['doc_mask'])
        for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
          score = score.item() # From tensor to value
          if self.debugPrint:
            print(score, did, docData[did])
          # Note that each element must be an array, b/c
          # we can generate more than one feature per document!
          sampleRet[did] = [score]

    if self.debugPrint:
      print('All scores:', sampleRet)

    return sampleRet

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Serving CEDR models.')

  parser.add_argument('--model', metavar='model type',
                      required=True, type=str,
                      help='type, e.g., vanilla_bert')

  parser.add_argument('--debug_print', action='store_true',
                      help='Provide debug output')

  parser.add_argument('--model_weights', metavar='model weights',
                      default=None, type=str,
                      help='model weight file')

  parser.add_argument('--batch_size', metavar='batch size',
                      default=DEFAULT_BATCH_SIZE, type=int,
                      help='batch size')

  parser.add_argument('--port', metavar='server port',
                      required=True, type=int,
                      help='Server port')

  parser.add_argument('--host', metavar='server host',
                      default='127.0.0.1', type=str,
                      help='server host addr to bind the port')

  args = parser.parse_args()


  multiThreaded=False #
  startQueryServer(args.host, args.port, multiThreaded, CedrQueryHandler(modelType=args.model,
                                                                        modelWeights=args.model_weights,
                                                                        batchSize=args.batch_size,
                                                                        debugPrint=args.debug_print))

