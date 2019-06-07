#!/usr/bin/env python
import glob
import sys
sys.path.append('gen-py')

from protocol.ExternalScorer import Processor
from protocol.ttypes import WordEntryInfo, TextEntryInfo, ScoringException

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

class BaseQueryHandler:
  def __init__(self):
    pass

  # One needs to override this function
  def getScores(self, query, docs):
    print('getScores', query, '# of docs', docs)
    sampleRet = {'id0' : 0.1, 'id1' : 0.2, 'id2' : 0.3}
    return sampleRet

if __name__ == '__main__':

  handler = BaseQueryHandler()

  processor = Processor(handler)
  transport = TSocket.TServerSocket(host='127.0.0.1', port=8080)
  tfactory = TTransport.TBufferedTransportFactory()
  pfactory = TBinaryProtocol.TBinaryProtocolFactory()

  #server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

  # You could do one of these for a multithreaded server
  server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
  # server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)

  print('Starting the server...')
  server.serve()
  print('done.')
