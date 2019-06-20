#!/usr/bin/env python3
import sys
sys.path.append('scripts/py_server')

from base_server import *

# Exclusive==True means that only one getScores
# function is executed at at time
class SampleQueryHandler(BaseQueryHandler):
  def __init__(self, exclusive=True):
    super().__init__(exclusive)

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
