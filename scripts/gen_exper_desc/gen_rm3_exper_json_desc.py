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
import os
import shutil

from flexneuart.gen_exper_desc import gen_rerank_descriptors, BaseParser, OUT_DIR_PARAM, REL_DESC_PATH_PARAM

MODEL_SRC_PATH = 'exper/sample_exper_desc/one_feat.model'
MODEL_DST_REL_PATH = 'models'
MODEL_DST_NAME = 'one_feat.model'


class ParserRM3Coeff(BaseParser):
    def init_add_args(self):
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

    def __init__(self, prog_name):
        super().__init__(prog_name)


parser = ParserRM3Coeff('RM3 tuning param generator')
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


class ExtrRM3GEN:
    def __call__(self):
        test_only = True
        for fid, extr_type in self.param_conf:
            yield fid, extr_type, test_only, model_rel_name

    def __init__(self, k1, b, index_field_name, query_field_name):
        self.param_conf = []

        param_arr = []

        for orig_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for top_doc_qty in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for top_term_qty in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    param_arr.append((orig_weight, top_doc_qty, top_term_qty))

        for orig_weight, top_doc_qty, top_term_qty in param_arr:
            bstr = '%g' % b
            k1str = '%g' % k1
            fid = f'rm3={index_field_name}+{query_field_name}_origWeight={orig_weight}_topDocQty={top_doc_qty}_topTermQty={top_term_qty}_k1={k1str}_{bstr}'

            extr_list = [{
                "type": "RM3Similarity",
                "params": {
                    "queryFieldName": query_field_name,
                    "indexFieldName": index_field_name,
                    "k1": k1str,
                    "b": bstr,
                    "origWeight": orig_weight,
                    "topDocQty": top_doc_qty,
                    "topTermQty": top_term_qty
                }}]

            self.param_conf.append((fid, extr_list))


if query_field_name is None:
    query_field_name = index_field_name

file_name_desc = '%s_%s' % (query_field_name, index_field_name)
prefix = f'rm3tune_{file_name_desc}'

gen_rerank_descriptors(args,
                       ExtrRM3GEN(k1=args.k1, b=args.b,
                                  index_field_name=index_field_name,
                                  query_field_name=query_field_name),
                     f'{prefix}.json', prefix)
