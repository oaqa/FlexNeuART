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
    Convert an NDRM model.
"""

import argparse
import torch
import os
from flexneuart.models.base import ModelSerializer, MODEL_PARAM_PREF
from flexneuart.utils import DictToObject

parser = argparse.ArgumentParser(description='Add doc2query fields to the existing JSONL data entries')

parser.add_argument('--input_dir', metavar='input directory',
                    help='input directory with IDFs, embeddings, and the model',
                    type=str, required=True)
parser.add_argument('--model_file', metavar='input model file',
                    type=str, default='model-dev.pt')
parser.add_argument('--model_type', metavar='input model type: ndrm1, ndrm2, ndrm3',
                    type=str, required=True)
parser.add_argument('--output', metavar='output model',
                    help='output model readable by FlexNeuART code',
                    type=str, required=True)
parser.add_argument('--max_query_len', metavar='max. query length',
                    type=int, required=True,
                    help='max. query length')
parser.add_argument('--dropout', metavar='droput', type=float, default=0.2)
parser.add_argument('--max_doc_len', metavar='max. document length',
                    type=int, required=True,
                    help='max. document length')

args = parser.parse_args()
print(args)

model_holder = ModelSerializer('ndrm')
input_dir = args.input_dir
model_type = args.model_type

print('Model type:', model_type, 'input directory:', input_dir)

model_args = {
    'max_query_len' : args.max_query_len,
    'max_doc_len' : args.max_doc_len,
    f'{MODEL_PARAM_PREF}embeddings' : os.path.join(input_dir, 'ndrm-embeddings.bin'),
    f'{MODEL_PARAM_PREF}idfs' : os.path.join(input_dir, 'ndrm-idfs.tsv'),
    f'{MODEL_PARAM_PREF}model_type' : model_type,
    f'{MODEL_PARAM_PREF}dropout' : args.dropout
}

print(model_args)

model_holder.create_model_from_args(DictToObject(model_args))

orig_model_fn = os.path.join(input_dir, args.model_file)
print('Loading model weights from:', orig_model_fn)
orig_model = torch.load(orig_model_fn, map_location='cpu')

# To be able to load weights, we need to update the state dictionary
state_dict_new = {f'model.{k}':v for k,v in orig_model['model_state_dict'].items()}

model : torch.nn.Module = model_holder.model
print(model.load_state_dict(state_dict_new, strict=True))

model_holder.save_all(args.output)
