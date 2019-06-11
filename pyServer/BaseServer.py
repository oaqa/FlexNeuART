import sys
sys.path.append('gen-py')

from protocol.ExternalScorer import Processor
from protocol.ttypes import WordEntryInfo, TextEntryInfo, ScoringException

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer


from threading import Lock

SAMPLE_HOST='127.0.0.1'
SAMPLE_PORT=8080

class BaseQueryHandler:
  def __init__(self, exclusive=True):
    self.lock_ = Lock() if exclusive else None 
  def getScores(self, query, docs):
    if self.lock_ is not None:
      with self.lock_:
        return self.computeScoresOverride(query, docs)
    else:
      return self.computeScoresOverride(query, docs)

  # This is the function to be implemented in the child class
  def computeScoresOverride(self, query, docs):
    raise NotImplementedError()


# This function starts the server and takes over the program control
def startQueryServer(host, port, multiThreaded, queryHandler):

  processor = Processor(queryHandler)

  transport = TSocket.TServerSocket(host=host, port=port)
  tfactory = TTransport.TBufferedTransportFactory()
  pfactory = TBinaryProtocol.TBinaryProtocolFactory()

  if multiThreaded:
    print('Starting a multi-threaded server...')
    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
  else:
    print('Starting a single-threaded server...')
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

  server.serve()
  print('done.')
