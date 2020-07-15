#!/usr/bin/env python
import sys
import argparse
import torch

sys.path.append('.')

from scripts.py_featextr_server.base_server import BaseQueryHandler, startQueryServer

import scripts.cedr.model_init_utils as model_init_utils

# CEDR imports

import scripts.cedr.train as train
import scripts.cedr.data as data

DEFAULT_BATCH_SIZE = 32

class CedrQueryHandler(BaseQueryHandler):
    # Exclusive==True means that only one getScores
    # function is executed at at time
    def __init__(self,
                    model,
                    modelWeights, batchSize, deviceName,
                    maxQueryLen, maxDocLen,
                    debugPrint=False):
        super().__init__(exclusive=True)

        self.debugPrint = debugPrint
        self.batchSize = batchSize

        self.maxQueryLen = maxQueryLen
        self.maxDocLen = maxDocLen
        self.deviceName = deviceName
        print('Maximum query/document len %d/%d device: %s' % (self.maxQueryLen, self.maxDocLen, self.deviceName))

        self.model = model
        self.model.to(self.deviceName)
        if modelWeights is not None:
            if self.debugPrint:
                print(f'Loading model {modelType} from {modelWeights}')
            self.model.load(modelWeights)

        # need to be in the eval mode
        self.model.eval()

    # This function needs to be overridden
    def computeScoresFromRawOverride(self, query, docs):
        if self.debugPrint:
            print('getScores', query.id, query.text)

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
                    for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                        score = score.item()  # From tensor to value
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

    model_init_utils.add_model_init_basic_args(parser)

    parser.add_argument('--debug_print', action='store_true',
                        help='Provide debug output')

    parser.add_argument('--model_weights', metavar='model weights',
                        default=None, type=str,
                        help='model weight file')

    parser.add_argument('--batch_size', metavar='batch size',
                        default=DEFAULT_BATCH_SIZE, type=int,
                        help='batch size')

    parser.add_argument('--port', metavar='server port',
                        required=True, type=int,
                        help='Server port')

    parser.add_argument('--host', metavar='server host',
                        default='127.0.0.1', type=str,
                        help='server host addr to bind the port')

    parser.add_argument('--device_name', metavar='CUDA device name or cpu', default='cuda:0',
                        help='The name of the CUDA device to use')

    parser.add_argument('--max_query_len', metavar='max. query length',
                        type=int, default=data.DEFAULT_MAX_QUERY_LEN,
                        help='max. query length')

    parser.add_argument('--max_doc_len', metavar='max. document length',
                        type=int, default=data.DEFAULT_MAX_DOC_LEN,
                        help='max. document length')

    args = parser.parse_args()

    model = model_init_utils.create_model_from_args(args)

    multiThreaded = False  #
    startQueryServer(args.host, args.port, multiThreaded, CedrQueryHandler(model=model,
                                                                           isBertLarge=args.bert_large,
                                                                           modelWeights=args.model_weights,
                                                                           batchSize=args.batch_size,
                                                                           debugPrint=args.debug_print,
                                                                           deviceName=args.device_name,
                                                                           maxQueryLen=args.max_query_len,
                                                                           maxDocLen=args.max_doc_len))
