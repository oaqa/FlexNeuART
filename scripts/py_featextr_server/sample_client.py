#!/usr/bin/env python

import sys
import glob

sys.path.append('scripts/py_featextr_server')
sys.path.append('scripts/py_featextr_server/gen-py')

from protocol.ExternalScorer import Client
from protocol.ttypes import WordEntryInfo, TextEntryParsed, TextEntryRaw, ScoringException

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from base_server import *

# Make socket
transport = TSocket.TSocket(SAMPLE_HOST, SAMPLE_PORT)

# Buffering is critical. Raw sockets are very slow
transport = TTransport.TBufferedTransport(transport)

# Wrap in a protocol
protocol = TBinaryProtocol.TBinaryProtocol(transport)

# Create a client to use the protocol encoder
client = Client(protocol)

# Connect!
transport.open()

query = TextEntryParsed("query_id", [])
docs = []
for did in ['1', '2', '3']:
  docs.append(TextEntryParsed(did, [WordEntryInfo(word="did: " + did, IDF=0.3, qty=3)]))

print('Calling using parsed text')
print(client.getScoresFromParsed(query, docs))


query = TextEntryRaw("query_id", "some query text")
docs = []
for did in ['1', '2', '3']:
  docs.append(TextEntryRaw(did, "some document text: " + did))

print('Calling using raw text')
print(client.getScoresFromRaw(query, docs))

