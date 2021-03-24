import sys

# Thrift files are generated from
# ./src/main/java/edu/cmu/lti/oaqa/flexneuart/letor/external/protocol.thrift
#

from scripts.py_featextr_server.python_generated.protocol.ExternalScorer import Processor
from scripts.py_featextr_server.python_generated.protocol.ttypes import ScoringException

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from threading import Lock

SAMPLE_HOST = '127.0.0.1'
SAMPLE_PORT = 8080


class BaseQueryHandler:
    def __init__(self, exclusive=True):
        self.lock_ = Lock() if exclusive else None
        if self.lock_ is not None:
            print('Locking the base server for single-threaded processing')
        else:
            print('NOT locking the base server for multi-threaded processing')

    # This function must remain in Camel-case, b/c it's tied to Java code
    def getScoresFromParsed(self, query, docs):
        try:
            if self.lock_ is not None:
                with self.lock_:
                    return self.compute_scores_from_parsed_override(query, docs)
            else:
                return self.compute_scores_from_parsed_override(query, docs)
        except Exception as e:
            raise ScoringException(str(e))

    # This function must remain in Camel-case, b/c it's tied to Java code
    def getScoresFromRaw(self, query, docs):
        try:
            if self.lock_ is not None:
                with self.lock_:
                    return self.compute_scores_from_raw_override(query, docs)
            else:
                return self.compute_scores_from_raw_override(query, docs)
        except Exception as e:
            raise ScoringException(str(e))

    def text_entry_to_str(self, te):
        arr = []
        for winfo in te.entries:
            arr.append('%s %g %d ' % (winfo.word, winfo.IDF, winfo.qty))
        return te.id + ' '.join(arr)

    def concat_text_entry_words(self, te):
        arr = [winfo.word for winfo in te.entries]
        return ' '.join(arr)

    # One or both functions need to be implemented in a child class
    def compute_scores_from_parsed_override(self, query, docs):
        raise ScoringException('Parsed fields are not supported by this server!')

    def compute_scores_from_raw_override(self, query, docs):
        raise ScoringException('Raw-text fields are not supported by this server!')


# This function starts the server and takes over the program control
def start_query_server(host, port, multi_threaded, query_handler):
    processor = Processor(query_handler)

    transport = TSocket.TServerSocket(host=host, port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    if multi_threaded:
        print('Starting a multi-threaded server...')
        server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    else:
        print('Starting a single-threaded server...')
        server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    server.serve()
    print('done.')
