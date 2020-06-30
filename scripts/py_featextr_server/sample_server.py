#!/usr/bin/env python
# This is a sample server that illustrates how to generate query-document scores
# and pass them to the client.
import sys

sys.path.append('.')

from scripts.py_featextr_server.base_server import BaseQueryHandler, startQueryServer, \
    SAMPLE_HOST, SAMPLE_PORT


# Exclusive==True means that only one getScores
# function is executed at at time
class SampleQueryHandler(BaseQueryHandler):
    def __init__(self, exclusive=True):
        super().__init__(exclusive)

    # This function needs to be overridden
    def computeScoresFromParsedOverride(self, query, docs):
        print('getScores', query.id, self.textEntryToStr(query))
        sampleRet = {}
        for e in docs:
            print(self.textEntryToStr(e))
            # Note that each element must be an array, b/c
            # we can generate more than one feature per document!
            sampleRet[e.id] = [0]
        return sampleRet

    # This function needs to be overridden
    def computeScoresFromRawOverride(self, query, docs):
        print('getScores', query.id, query.text)
        sampleRet = {}
        for e in docs:
            print(e.text)
            # Note that each element must be an array, b/c
            # we can generate more than one feature per document!
            sampleRet[e.id] = [0]
        return sampleRet


if __name__ == '__main__':
    multiThreaded = True
    startQueryServer(SAMPLE_HOST, SAMPLE_PORT, multiThreaded, SampleQueryHandler(exclusive=False))
