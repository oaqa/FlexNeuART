#!/usr/bin/env python3
import sys
sys.path.append('gen-py')

from BaseServer import *

# Exclusive==True means that only one getScores
# function is executed at at time
class SampleQueryHandler(BaseQueryHandler):
  def __init__(self, exclusive=True):
    super().__init__(exclusive)

  def textEntryToStr(self, te):
    arr=[]
    for winfo in te.entries:
     arr.append('%s %g %d ' % (winfo.word, winfo.IDF, winfo.qty))
    return te.id + ' '.join(arr)

  # This function needs to be overriden
  def computeScoresOverride(self, query, docs):
    print('getScores', self.textEntryToStr(query))
    sampleRet = {}
    for e in docs:
      print(self.textEntryToStr(e))
      sampleRet[e.id] = [0]
    return sampleRet

if __name__ == '__main__':

  multiThreaded=True
  startQueryServer(SAMPLE_HOST, SAMPLE_PORT, multiThreaded, SampleQueryHandler(exclusive=False))
