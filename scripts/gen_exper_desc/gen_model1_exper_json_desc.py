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
import os, sys, json, re

sys.path.append('.')

from scripts.gen_exper_desc.common_gen_desc import gen_rerank_descriptors, BaseParser
from scripts.config import TEXT_FIELD_NAME

class ExtrModel1JsonGEN:

    def __call__(self):
        test_only = False
        for fid, extr_type in self.param_conf:
            yield fid, extr_type, test_only, None

    def __init__(self, k1, b,
                 index_field_name, query_field_name,
                 text_field_name=TEXT_FIELD_NAME):
        self.k1 = k1
        self.b = b
        self.query_field_name = query_field_name
        self.index_field_name = index_field_name

        self.param_conf = []

        param_arr = []

        for probSelfTran in [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75]:
            for lamb in [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5]:
                param_arr.append((probSelfTran, lamb))

        param_arr.append((0.6, 0.05))
        param_arr.append((0.7, 0.05))
        param_arr.append((0.8, 0.05))
        param_arr.append((0.9, 0.05))

        param_arr.append((0.9, 0.01))
        param_arr.append((0.9, 0.001))
        param_arr.append((0.9, 0.0001))

        for probSelfTran, lamb in param_arr:
            fid = f'bm25={text_field_name}+model1={index_field_name}+lambda=%g+probSelfTran=%g' % (lamb, probSelfTran)

            extr_list = [
                {
                    "type": "Model1Similarity",
                    "params": {
                        "queryFieldName": self.query_field_name,
                        "indexFieldName": self.index_field_name,
                        "gizaIterQty": "5",
                        "probSelfTran": probSelfTran,
                        "lambda": lamb,
                        "minModel1Prob": "2.5e-3f"
                    }
                },
                {
                    "type": "TFIDFSimilarity",
                    "params": {
                        "indexFieldName": text_field_name,
                        "similType": "bm25",
                        "k1": self.k1,
                        "b": self.b
                    }
                }
            ]

            self.param_conf.append((fid, extr_list))

        param_arr = []
        param_arr.append((0.9, 0.00001, 1e-3))
        param_arr.append((0.9, 0.00001, 1e-4))
        param_arr.append((0.9, 0.00001, 5e-4))
        param_arr.append((0.9, 0.00001, 2.5e-4))

        for probSelfTran, lamb, min_model1_prob in param_arr:
            fid = f'bm25={text_field_name}+model1={index_field_name}+lambda=%g+probSelfTran=%g+minTranProb=%g' % (
                lamb, probSelfTran, min_model1_prob)

            extr_list = [
                {
                    "type": "Model1Similarity",
                    "params": {
                        "queryFieldName": self.query_field_name,
                        "indexFieldName": self.index_field_name,
                        "gizaIterQty": "5",
                        "probSelfTran": str(probSelfTran) + "f",  # for float in Java
                        "lambda": lamb,
                        "minModel1Prob": min_model1_prob
                    }
                },
                {
                    "type": "TFIDFSimilarity",
                    "params": {
                        "indexFieldName": index_field_name,
                        "similType": "bm25",
                        "k1": self.k1,
                        "b": self.b
                    }
                }
            ]

            self.param_conf.append((fid, extr_list))


class ParserWithModel1Coeff(BaseParser):
    def init_add_args(self):
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

    def __init__(self, prog_name):
        super().__init__(prog_name)


parser = ParserWithModel1Coeff('Model1 tuning param generator')
parser.parse_args()
args = parser.get_args()
index_field_name = args.index_field_name
query_field_name = args.query_field_name
file_name_desc = '%s_%s' % (query_field_name, index_field_name)
model1prefix = f'model1tune_{file_name_desc}'
gen_rerank_descriptors(args,
                       ExtrModel1JsonGEN(k1=args.k1, b=args.b,
                                         index_field_name=index_field_name,
                                         query_field_name=query_field_name),
                       model1prefix + '.json', model1prefix)
