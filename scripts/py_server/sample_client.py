#!/usr/bin/env python

import sys
import glob
sys.path.append('gen-py')

from protocol.ExternalScorer import Client
from protocol.ttypes import WordEntryInfo, TextEntryInfo, ScoringException

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from BaseServer import *

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

query = TextEntryInfo("query_id", [])
docs = []
for did in ['1', '2', '3']:
  docs.append(TextEntryInfo(did, []))

print(client.getScores(query, docs))

