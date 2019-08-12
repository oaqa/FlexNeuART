#!/usr/bin/env python3
import sys
import argparse

sys.path.append('scripts/py_server')
sys.path.append('scripts/data')

from base_server import *

import numpy as np

from scipy.spatial.distance import cosine

from base_server import *

from utils import loadEmbeddings, createEmbedMap, robustCosineSimil

# Exclusive==True means that only one getScores
# function is executed at at time
class CosineSimilQueryHandler(BaseQueryHandler):
  def __init__(self, queryEmbedFile, docEmbedFile, exclusive, debugPrint=False, useIDF=True):
    super().__init__(exclusive)

    self.debugPrint = debugPrint
    self.useIDF = useIDF

    print('Loading answer embeddings from: ' + docEmbedFile)
    answWords, self.answEmbed = loadEmbeddings(docEmbedFile)
    self.answEmbedMap = createEmbedMap(answWords)

    if queryEmbedFile is not None:
      print('Loading query embeddings from: ' + queryEmbedFile)
      queryWords, self.queryEmbed = loadEmbeddings(queryEmbedFile)
      self.queryEmbedMap = createEmbedMap(queryWords)
    else:
      self.queryEmbed = self.answEmbed
      self.queryEmbedMap = self.answEmbedMap
    print('Loading is done!')

  def textEntryToStr(self, te):
    arr=[]
    if self.debugPrint:
      for winfo in te.entries:
        arr.append('%s %g %d ' % (winfo.word, winfo.IDF, winfo.qty))
    return 'docId='+te.id + ' ' + ' '.join(arr)

  def createDocEmbed(self, isQuery, textEntry):

    if isQuery:
      embeds = self.queryEmbed
      embedMap = self.queryEmbedMap
    else:
      embeds = self.answEmbed
      embedMap = self.answEmbedMap

    zerov = np.zeros_like(embeds[0])
    res = zerov

    for winfo in textEntry.entries:
      vectMult =  winfo.qty
      if self.useIDF:
        vectMult *= winfo.IDF
      word = winfo.word
      if word in embedMap:
        res += embeds[embedMap[word]] * vectMult

    return res


  # This function overrids the parent class
  def computeScoresOverride(self, query, docs):
    if self.debugPrint:
      print('getScores', self.textEntryToStr(query))
    ret = {}
    queryEmbed = self.createDocEmbed(True, query)
    if self.debugPrint:
      print(queryEmbed)
    for d in docs:
      if self.debugPrint:
        print(self.textEntryToStr(d))
      docEmbed = self.createDocEmbed(False, d)
      if self.debugPrint:
        print(docEmbed)
      # Regular cosine deals poorly with all-zero vectors
      simil=robustCosineSimil(docEmbed, queryEmbed)
      #simil = (1-cosine(docEmbed, queryEmbed))
      ret[d.id] = [simil]

    return ret

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Serving word-embedding models.')

  parser.add_argument('--query_embed', metavar='query embeddings',
                      default=None, type=str,
                      help='Optional query embeddings file')

  parser.add_argument('--doc_embed', metavar='doc embeddings',
                      required=True, type=str,
                      help='document embeddings file')

  parser.add_argument('--debug_print', action='store_true',
                      help='Provide debug output')

  parser.add_argument('--port', metavar='server port',
                      required=True, type=int,
                      help='Server port')

  parser.add_argument('--host', metavar='server host',
                      default='127.0.0.1', type=str,
                      help='server host addr to bind the port')

  args = parser.parse_args()

  multiThreaded=True
  startQueryServer(args.host, args.port, multiThreaded, 
                    CosineSimilQueryHandler(exclusive=False, 
                                            queryEmbedFile=args.query_embed,
                                            docEmbedFile=args.doc_embed,
                                            debugPrint=args.debug_print))
