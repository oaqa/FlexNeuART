#!/usr/bin/env python
import sys
import argparse
import torch
import time

sys.path.append('.')

from scripts.py_featextr_server.python_generated.protocol.ttypes import TextEntryRaw
from scripts.py_featextr_server.base_server import BaseQueryHandler, startQueryServer

import scripts.cedr.model_init_utils as model_init_utils
import scripts.cedr.data as data

DEFAULT_BATCH_SIZE = 32

class CedrQueryHandler(BaseQueryHandler):
    # Exclusive==True means that only one getScores
    # function is executed at at time
    def __init__(self,
                    model,
                    batchSize, deviceName,
                    maxQueryLen, maxDocLen,
                    exclusive,
                    debugPrint=False):
        super().__init__(exclusive=exclusive)

        self.debugPrint = debugPrint
        self.batchSize = batchSize

        self.maxQueryLen = maxQueryLen
        self.maxDocLen = maxDocLen
        self.deviceName = deviceName
        print('Maximum query/document len %d/%d device: %s' % (self.maxQueryLen, self.maxDocLen, self.deviceName))

        self.model = model

        self.model.to(self.deviceName)

        # need to be in the eval mode
        self.model.eval()

    def computeScoresFromParsedOverride(self, query, docs):
        queryRaw = TextEntryRaw(query.id, self.concatTextEntryWords(query))
        docsRaw = []
        for e in docs:
            docsRaw.append(TextEntryRaw(e.id, self.concatTextEntryWords(e)))

        return self.computeScoresFromRawOverride(queryRaw, docsRaw)

    def computeScoresFromRawOverride(self, query, docs):
        print('Processing query:', query.id, query.text, '# of docs: ', len(docs))

        queryData = {query.id: query.text}
        # Run maps queries to arrays of document IDs see iter_valid_records (train.py)
        run = {query.id: [e.id for e in docs]}

        docData = {}
        for e in docs:
            docData[e.id] = e.text

        sampleRet = {}

        if docData:

            # based on the code from run_model function (train.py)
            dataSet = queryData, docData
            # must disable gradient computation to greatly reduce memory requirements and speed up things
            with torch.no_grad():
                for records in data.iter_valid_records(self.model, self.deviceName, dataSet, run,
                                                       self.batchSize,
                                                       self.maxQueryLen, self.maxDocLen):
                    scores = self.model(records['query_tok'],
                                        records['query_mask'],
                                        records['doc_tok'],
                                        records['doc_mask'])

                    # tolist() works much faster compared to extracting scores
                    # one by one using .item()
                    scores = scores.tolist()

                    for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                        if self.debugPrint:
                            print(score, did, docData[did])
                        # Note that each element must be an array, b/c
                        # we can generate more than one feature per document!
                        sampleRet[did] = [score]

        if self.debugPrint:
            print('All scores:', sampleRet)

        return sampleRet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serving CEDR models.')

    model_init_utils.add_model_init_basic_args(parser, False)

    parser.add_argument('--debug_print', action='store_true',
                        help='Provide debug output')

    parser.add_argument('--batch_size', metavar='batch size',
                        default=DEFAULT_BATCH_SIZE, type=int,
                        help='batch size')

    parser.add_argument('--port', metavar='server port',
                        required=True, type=int,
                        help='Server port')

    parser.add_argument('--host', metavar='server host',
                        default='127.0.0.1', type=str,
                        help='server host addr to bind the port')


    args = parser.parse_args()

    if args.init_model is None:
        if args.model is not None and args.init_model_weights is not None:
            model = model_init_utils.create_model_from_args(args)
            print('Loading model weights from:', args.init_model_weights.name)
            # If we load weights here, we must set strict to True:
            # this would prevent accidental loading of partial models.
            # Partial models are sure fine to load during training (as a reasonable
            # initialization), but not during test time.
            model.load_state_dict(torch.load(args.init_model_weights.name, map_location='cpu'),
                                  strict=True)
        else:
            print('Specify the model file: --init_model or model type and model weights')
            sys.exit(1)
    else:
        model = torch.load(args.init_model, map_location='cpu')

    multiThreaded = False  # if we set to True, we can often run out of CUDA memory.
    startQueryServer(args.host, args.port, multiThreaded, CedrQueryHandler(model=model,
                                                                           batchSize=args.batch_size,
                                                                           debugPrint=args.debug_print,
                                                                           deviceName=args.device_name,
                                                                           maxQueryLen=args.max_query_len,
                                                                           maxDocLen=args.max_doc_len,
                                                                           exclusive=not multiThreaded))
