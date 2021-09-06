#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# This is a sample client that retrieves query-document scores from a sample server

import sys

sys.path.append('.')

from scripts.featextr_server.python_generated.protocol.ExternalScorer import Client
from scripts.featextr_server.python_generated.protocol.ttypes import \
    WordEntryInfo, TextEntryParsed, TextEntryRaw

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from scripts.featextr_server.base_server import SAMPLE_PORT, SAMPLE_HOST

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
