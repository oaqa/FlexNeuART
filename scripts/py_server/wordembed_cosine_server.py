#!/usr/bin/env python3
import sys
sys.path.append('gen-py')

import numpy as np

from scipy.spatial.distance import cosine

from BaseServer import *

from utils import loadEmbeddings, createEmbedMap, robustCosineSimil

DEBUG_PRINT=False
USE_IDF=False

# Exclusive==True means that only one getScores
# function is executed at at time
class CosineSimilQueryHandler(BaseQueryHandler):
  def __init__(self, exclusive=False):
    super().__init__(exclusive)

    print('Loading query embeddings')
    queryWords, self.queryEmbed = loadEmbeddings('../WordEmbeddings/manner/starspace_unlemm.query')
    self.queryEmbedMap = createEmbedMap(queryWords)
    print('Loading answer embeddings')
    answWords, self.answEmbed = loadEmbeddings('../WordEmbeddings/manner/starspace_unlemm.answer')
    self.answEmbedMap = createEmbedMap(answWords)

  def textEntryToStr(self, te):
    arr=[]
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
      if USE_IDF:
        vectMult *= winfo.IDF
      word = winfo.word
      if word in embedMap:
        res += embeds[embedMap[word]] * vectMult

    return res


  # This function overrids the parent class
  def computeScoresOverride(self, query, docs):
    if DEBUG_PRINT:
      print('getScores', self.textEntryToStr(query))
    ret = {}
    queryEmbed = self.createDocEmbed(True, query)
    if DEBUG_PRINT:
      print(queryEmbed)
    for d in docs:
      if DEBUG_PRINT:
        print(self.textEntryToStr(d))
      docEmbed = self.createDocEmbed(False, d)
      if DEBUG_PRINT:
        print(docEmbed)
      # Regular cosine deals poorly with all-zero vectors
      simil=robustCosineSimil(docEmbed, queryEmbed)
      #simil = (1-cosine(docEmbed, queryEmbed))
      ret[d.id] = [simil]

    return ret

if __name__ == '__main__':

  multiThreaded=True
  startQueryServer(SAMPLE_HOST, SAMPLE_PORT, multiThreaded, CosineSimilQueryHandler(exclusive=False))
