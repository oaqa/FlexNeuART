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
"""
    Convert a COLBERT model trained using original COLBERT code to FlexNeuART format.
    We make an assumption that the model is based on BERT base model or comparable Transformer.

    COLBERT code: https://github.com/stanford-futuredata/ColBERT/tree/new_api/colbert
"""

import argparse
import torch

from flexneuart.models.base import ModelSerializer, MODEL_PARAM_PREF
from flexneuart.config import BERT_BASE_MODEL
from flexneuart.utils import DictToObject
from flexneuart.models.colbert.colbert_wrapper import INNER_MODEL_ATTR
from flexneuart.models.colbert.colbert import ColBERT


parser = argparse.ArgumentParser(description='Convert COLBERT model to FlexNeuART format')

parser.add_argument('--input_dir', metavar='input directory',
                    help='input directory the saved checkpoint',
                    type=str, required=True)
parser.add_argument('--output', metavar='output model',
                    help='output model readable by FlexNeuART code',
                    type=str, required=True)

args = parser.parse_args()
print(args)

model_holder = ModelSerializer('colbert')
input_dir = args.input_dir
bert_flavor = BERT_BASE_MODEL

print('BERT flavor:', bert_flavor, 'input directory:', input_dir)

orig_colbert = ColBERT(bert_flavor=bert_flavor, colbert_config=args.input_dir)
colbert_config = orig_colbert.colbert_config

# note that we need to memorize original colbert lengths for the query and document
model_args = {
    'max_query_len' : colbert_config.query_maxlen,
    'max_doc_len' : colbert_config.doc_maxlen,
    f'{MODEL_PARAM_PREF}bert_flavor' : bert_flavor,
    f'{MODEL_PARAM_PREF}colbert_config' : colbert_config
}

print(model_args)

# create a randomly initialized model
model_holder.create_model_from_args(DictToObject(model_args))
# transfer model weights
inner_model : torch.nn.Module = getattr(model_holder.model, INNER_MODEL_ATTR)
print(inner_model.load_state_dict(orig_colbert.state_dict()), strict=True)

model_holder.save_all(args.output)


