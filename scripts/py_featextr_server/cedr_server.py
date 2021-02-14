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
                    modelList, modelWeightList,
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

        self.modelList = modelList
        for model in self.modelList:
            model.to(self.deviceName)
            # need to be in the eval mode
            model.eval()

        # This covers both None and empty list
        if not modelWeightList:
            modelWeightList = [1.0] * len(modelList)

        assert len(modelWeightList) == len(modelList), "The weight list should be empty or have the same # of elements as the model list!"
        # Normalize weights
        weightSum = sum(modelWeightList)
        assert weightSum > 0, "The sum of the weights should be positive!"

        self.modelWeightList = [float(w) / weightSum for w in modelWeightList]
        print('Normalized model weights:', self.modelWeightList)


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
                for model, weight in zip(self.modelList, self.modelWeightList):
                    for records in data.iter_valid_records(model, self.deviceName, dataSet, run,
                                                           self.batchSize,
                                                           self.maxQueryLen, self.maxDocLen):

                        scores = model(records['query_tok'],
                                        records['query_mask'],
                                        records['doc_tok'],
                                        records['doc_mask'])


                        # tolist() works much faster compared to extracting scores
                        # one by one using .item()
                        scores = (weight * scores).tolist()

                        for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                            if self.debugPrint:
                                print(score, did, docData[did])
                            # Note that each element must be an array, b/c
                            # one can potentially generate more than one feature per document!
                            if did in sampleRet:
                                sampleRet[did][0] += score
                            else:
                                sampleRet[did] = [score]

        if self.debugPrint:
            print('All scores:', sampleRet)

        return sampleRet


def add_eval_model_init_args(parser):

    parser.add_argument('--model', metavar='model',
                        help='a model to use: ' + ' '.join(list(model_init_utils.MODEL_MAP.keys())),
                        choices=model_init_utils.MODEL_MAP.keys(), default='vanilla_bert')

    parser.add_argument('--init_model_weights',
                        metavar='model weights', help='initial model weights',
                        type=argparse.FileType('rb'),
                        nargs='+',
                        default=None)

    parser.add_argument('--init_model',
                        metavar='initial model',
                        help='initial *COMPLETE* model with heads and extra parameters',
                        type=argparse.FileType('rb'),
                        nargs='+',
                        default=None)

    parser.add_argument('--model_mix_weights',
                        metavar='model mixing weights',
                        help='weights to linearly combine model scores (same weights by default)',
                        type=float,
                        nargs='+',
                        default=None)

    parser.add_argument('--max_query_len', metavar='max. query length',
                        type=int, default=data.DEFAULT_MAX_QUERY_LEN,
                        help='max. query length')

    parser.add_argument('--max_doc_len', metavar='max. document length',
                        type=int, default=data.DEFAULT_MAX_DOC_LEN,
                        help='max. document length')


    parser.add_argument('--device_name', metavar='CUDA device name or cpu', default='cuda:0',
                        help='The name of the CUDA device to use')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serving CEDR models.')

    add_eval_model_init_args(parser)

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

    model_list = []

    assert(args.init_model is None or type(args.init_model) == list)
    assert (args.init_model_weights is None or type(args.init_model_weights) == list)
    assert (args.model_mix_weights is None or type(args.model_mix_weights) == list)

    modelList = []

    if args.init_model is None:
        # TODO this one isn't properly tested
        if args.model is not None and args.init_model_weights is not None:
            for fname in args.init_model_weights.name:
                model = model_init_utils.create_model_from_args(args)
                print('Loading model weights from:', fname.name)
                # If we load weights here, we must set strict to True:
                # this would prevent accidental loading of partial models.
                # Partial models are sure fine to load during training (as a reasonable
                # initialization), but not during test time.
                model.load_state_dict(torch.load(fname.name, map_location='cpu'),
                                      strict=True)
                modelList.append(model)
        else:
            print('Specify the model file: --init_model or model type and model weights')
            sys.exit(1)
    else:
        for fname in args.init_model:
            print('Loading model from:', fname.name)
            model = torch.load(fname.name, map_location='cpu')
            modelList.append(model)

    multiThreaded = False  # if we set to True, we can often run out of CUDA memory.
    startQueryServer(args.host, args.port, multiThreaded, CedrQueryHandler(modelList=modelList,
                                                                           modelWeightList=args.model_mix_weights,
                                                                           batchSize=args.batch_size,
                                                                           debugPrint=args.debug_print,
                                                                           deviceName=args.device_name,
                                                                           maxQueryLen=args.max_query_len,
                                                                           maxDocLen=args.max_doc_len,
                                                                           exclusive=not multiThreaded))
