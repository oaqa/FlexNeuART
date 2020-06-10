#!/usr/bin/env python
import os
import sys
import json
import re
import shutil

sys.path.append('scripts')

from data_convert.convert_common import *
from gen_exper_desc.common_gen_desc import *

MODEL_SRC_PATH = 'scripts/exper/sample_exper_desc/one_feat.model'
MODEL_DST_REL_PATH = 'models'
MODEL_DST_NAME = 'one_feat.model'


class ParserRM3Coeff(BaseParser):
    def initAddArgs(self):
        self.parser.add_argument('-b', metavar='BM25 b',
                                 help='BM25 parameter b',
                                 type=float, required=True)
        self.parser.add_argument('-k1', metavar='BM25 k1',
                                 help='BM25 parameter b',
                                 type=float, required=True)
        self.parser.add_argument('--index_field_name',
                                 metavar='BM25 index field name',
                                 help='an index field for BM25 score', required=True)
        self.parser.add_argument('--query_field_name',
                                 metavar='BM25 query field name',
                                 help='an query field for BM25 score', default=None)

    def __init__(self, progName):
        super().__init__(progName)


parser = ParserRM3Coeff('RM3 tuning param generator')
parser.parseArgs()
args = parser.getArgs()
args_var = vars(args)

indexFieldName = args.index_field_name
queryFieldName = args.query_field_name
outdir = args_var[OUT_DIR_PARAM]
outModelDir = os.path.join(outdir, MODEL_DST_REL_PATH)
if not os.path.exists(outModelDir):
    os.makedirs(outModelDir)
shutil.copyfile(MODEL_SRC_PATH, os.path.join(outModelDir, MODEL_DST_NAME))

modelRelName = os.path.join(args_var[REL_DESC_PATH_PARAM], MODEL_DST_REL_PATH, MODEL_DST_NAME)


class ExtrRM3GEN:
    def __call__(self):
        testOnly = True
        for fid, extrType in self.paramConf:
            yield fid, extrType, testOnly, modelRelName

    def __init__(self, k1, b, indexFieldName, queryFieldName):
        self.paramConf = []

        paramArr = []

        for origWeight in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for topDocQty in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for topTermQty in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    paramArr.append((origWeight, topDocQty, topTermQty))

        for origWeight, topDocQty, topTermQty in paramArr:
            bstr = '%g' % b
            k1str = '%g' % k1
            fid = f'rm3={indexFieldName}+{queryFieldName}_origWeight={origWeight}_topDocQty={topDocQty}_topTermQty={topTermQty}_k1={k1str}_{bstr}'

            extrList = [{
                "type": "RM3Similarity",
                "params": {
                    "queryFieldName": queryFieldName,
                    "indexFieldName": indexFieldName,
                    "k1": k1str,
                    "b": bstr,
                    "origWeight": origWeight,
                    "topDocQty": topDocQty,
                    "topTermQty": topTermQty
                }}]

            self.paramConf.append((fid, {"extractors": extrList}))


if queryFieldName is None:
    queryFieldName = indexFieldName

fileNameDesc = '%s_%s' % (queryFieldName, indexFieldName)
prefix = f'rm3tune_{fileNameDesc}'

genRerankDescriptors(args,
                     ExtrRM3GEN(k1=args.k1, b=args.b, indexFieldName=indexFieldName, queryFieldName=queryFieldName),
                     f'{prefix}.json', prefix)
