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
import sys
import os
import shutil

sys.path.append('.')

from scripts.gen_exper_desc.common_gen_desc import gen_rerank_descriptors, BaseParser, OUT_DIR_PARAM, REL_DESC_PATH_PARAM

MODEL_SRC_PATH = 'scripts/exper/sample_exper_desc/one_feat.model'
MODEL_DST_REL_PATH = 'models'
MODEL_DST_NAME = 'one_feat.model'


class ParserBM25Coeff(BaseParser):
    def init_add_args(self):
        self.parser.add_argument('--index_field_name',
                                 metavar='BM25 index field name',
                                 help='an index field for BM25 score', required=True)
        self.parser.add_argument('--query_field_name',
                                 metavar='BM25 query field name',
                                 help='an query field for BM25 score', default=None)


parser = ParserBM25Coeff('BM25 tuning param generator')
parser.parse_args()

args = parser.get_args()
args_var = vars(args)

index_field_name = args.index_field_name
query_field_name = args.query_field_name
outdir = args_var[OUT_DIR_PARAM]
out_model_dir = os.path.join(outdir, MODEL_DST_REL_PATH)
if not os.path.exists(out_model_dir):
    os.makedirs(out_model_dir)
shutil.copyfile(MODEL_SRC_PATH, os.path.join(out_model_dir, MODEL_DST_NAME))

model_rel_name = os.path.join(args_var[REL_DESC_PATH_PARAM], MODEL_DST_REL_PATH, MODEL_DST_NAME)

if query_field_name is None:
    query_field_name = index_field_name


class ExtrBM25JsonGEN:
    def __init__(self, index_field_name, query_field_name):
        self.index_field_name = index_field_name
        self.query_field_name = query_field_name

    def __call__(self):

        for bi in range(8):
            for k1i in range(7):
                b = 0.3 + 0.1 * bi
                # b should be between 0 and 1
                assert b <= 1
                k1 = 0.4 + 0.2 * k1i
                bstr = '%g' % b
                k1str = '%g' % k1
                fid = 'bm25tune_k1=%s_b=%s' % (k1str, bstr)

                json_desc = {
                    "extractors": [
                        {
                            "type": "TFIDFSimilarity",
                            "params": {
                                "queryFieldName": self.query_field_name,
                                "indexFieldName": self.index_field_name,
                                "similType": "bm25",
                                "k1": k1str,
                                "b": bstr
                            }
                        }
                    ]
                }

                # Test only is true, b/c there's nothing to train, but we need to provide the model
                test_only = True
                yield fid, json_desc, test_only, model_rel_name


file_name_desc = '%s_%s' % (query_field_name, index_field_name)
prefix = f'bm25tune_{file_name_desc}'
gen_rerank_descriptors(args, ExtrBM25JsonGEN(index_field_name, query_field_name),
                     f'{prefix}.json', prefix)
