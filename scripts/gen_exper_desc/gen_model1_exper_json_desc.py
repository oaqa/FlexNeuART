#!/usr/bin/env python
import os, sys, json, re

sys.path.append('.')

from scripts.gen_exper_desc.common_gen_desc import genRerankDescriptors, BaseParser
from scripts.config import TEXT_FIELD_NAME

class ExtrModel1JsonGEN:

    def __call__(self):
        testOnly = False
        for fid, extrType in self.paramConf:
            yield fid, extrType, testOnly, None

    def __init__(self, k1, b,
                 indexFieldName, queryFieldName,
                 textFieldName=TEXT_FIELD_NAME):
        self.k1 = k1
        self.b = b
        self.queryFieldName = queryFieldName
        self.indexFieldName = indexFieldName

        self.paramConf = []

        paramArr = []

        for probSelfTran in [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75]:
            for lamb in [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5]:
                paramArr.append((probSelfTran, lamb))

        paramArr.append((0.6, 0.05))
        paramArr.append((0.7, 0.05))
        paramArr.append((0.8, 0.05))
        paramArr.append((0.9, 0.05))

        paramArr.append((0.9, 0.01))
        paramArr.append((0.9, 0.001))
        paramArr.append((0.9, 0.0001))

        for probSelfTran, lamb in paramArr:
            fid = f'bm25={textFieldName}+model1={indexFieldName}+lambda=%g+probSelfTran=%g' % (lamb, probSelfTran)

            extrList = [
                {
                    "type": "Model1Similarity",
                    "params": {
                        "queryFieldName": self.queryFieldName,
                        "indexFieldName": self.indexFieldName,
                        "gizaIterQty": "5",
                        "probSelfTran": probSelfTran,
                        "lambda": lamb,
                        "minModel1Prob": "2.5e-3f"
                    }
                },
                {
                    "type": "TFIDFSimilarity",
                    "params": {
                        "indexFieldName": textFieldName,
                        "similType": "bm25",
                        "k1": self.k1,
                        "b": self.b
                    }
                }
            ]

            self.paramConf.append((fid, {"extractors": extrList}))

        paramArr = []
        paramArr.append((0.9, 0.00001, 1e-3))
        paramArr.append((0.9, 0.00001, 1e-4))
        paramArr.append((0.9, 0.00001, 5e-4))
        paramArr.append((0.9, 0.00001, 2.5e-4))

        for probSelfTran, lamb, minModel1Prob in paramArr:
            fid = f'bm25={textFieldName}+model1={indexFieldName}+lambda=%g+probSelfTran=%g+minTranProb=%g' % (
                lamb, probSelfTran, minModel1Prob)

            extrList = [
                {
                    "type": "Model1Similarity",
                    "params": {
                        "queryFieldName": self.queryFieldName,
                        "indexFieldName": self.indexFieldName,
                        "gizaIterQty": "5",
                        "probSelfTran": str(probSelfTran) + "f",  # for float in Java
                        "lambda": lamb,
                        "minModel1Prob": minModel1Prob
                    }
                },
                {
                    "type": "TFIDFSimilarity",
                    "params": {
                        "indexFieldName": indexFieldName,
                        "similType": "bm25",
                        "k1": self.k1,
                        "b": self.b
                    }
                }
            ]

            self.paramConf.append((fid, {"extractors": extrList}))


class ParserWithModel1Coeff(BaseParser):
    def initAddArgs(self):
        self.parser.add_argument('-b', metavar='BM25 b',
                                 help='BM25 parameter b',
                                 type=float, required=True)
        self.parser.add_argument('-k1', metavar='BM25 k1',
                                 help='BM25 parameter b',
                                 type=float, required=True)
        self.parser.add_argument('--index_field_name',
                                 metavar='BITEXT index field name',
                                 help='an index field for BM25 score', required=True)
        self.parser.add_argument('--query_field_name',
                                 metavar='BITEXT query field name',
                                 help='an query field for BM25 score', default=None)

    def __init__(self, progName):
        super().__init__(progName)


parser = ParserWithModel1Coeff('Model1 tuning param generator')
parser.parseArgs()
args = parser.getArgs()
indexFieldName = args.index_field_name
queryFieldName = args.query_field_name
fileNameDesc = '%s_%s' % (queryFieldName, indexFieldName)
model1prefix = f'model1tune_{fileNameDesc}'
genRerankDescriptors(args,
                     ExtrModel1JsonGEN(k1=args.k1, b=args.b,
                                       indexFieldName=indexFieldName, queryFieldName=queryFieldName),
                     model1prefix + '.json', model1prefix)
