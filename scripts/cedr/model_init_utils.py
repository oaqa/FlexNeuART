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
import inspect
import argparse

from scripts.config import BERT_BASE_MODEL

import scripts.cedr.data as data
import scripts.cedr.modeling_basic as modeling_basic
import scripts.cedr.modeling_cedr as modeling_cedr


VANILLA_BERT = 'vanilla_bert'

MODEL_MAP = {
    VANILLA_BERT: modeling_basic.VanillaBertRanker,
    'cedr_pacrr': modeling_cedr.CedrPacrrRanker,
    'cedr_knrm': modeling_cedr.CedrKnrmRanker,
    'cedr_drmm': modeling_cedr.CedrDrmmRanker
}

def add_model_init_basic_args(parser, add_train_params):

    parser.add_argument('--model', metavar='model',
                        help='a model to use: ' + ' '.join(list(MODEL_MAP.keys())),
                        choices=MODEL_MAP.keys(), default='vanilla_bert')

    parser.add_argument('--init_model_weights',
                        metavar='model weights', help='initial model weights',
                        type=argparse.FileType('rb'), default=None)

    parser.add_argument('--init_model',
                        metavar='initial model',
                        help='initial *COMPLETE* model with heads and extra parameters',
                        type=argparse.FileType('rb'), default=None)

    parser.add_argument('--max_query_len', metavar='max. query length',
                        type=int, default=data.DEFAULT_MAX_QUERY_LEN,
                        help='max. query length')

    parser.add_argument('--max_doc_len', metavar='max. document length',
                        type=int, default=data.DEFAULT_MAX_DOC_LEN,
                        help='max. document length')


    parser.add_argument('--device_name', metavar='CUDA device name or cpu', default='cuda:0',
                        help='The name of the CUDA device to use')


    if add_train_params:

        parser.add_argument(f'--{MODEL_PARAM_PREF}dropout', type=float,
                            default=modeling_basic.DEFAULT_BERT_DROPOUT,
                            metavar='optional model droput',
                            help='optional model droput (not for every model)')

        parser.add_argument(f'--{MODEL_PARAM_PREF}bert_flavor',
                            type=str, default=BERT_BASE_MODEL,
                            metavar=f'BERT model name',
                            help='A name of the pre-trained BERT model, e.g., {BERT_BASE_MODEL}')

        parser.add_argument(f'--{MODEL_PARAM_PREF}vocab_file',
                            metavar='vocabulary file',
                            type=str, default=None,
                            help='a previously created vocabulary file')

        parser.add_argument(f'--{MODEL_PARAM_PREF}use_fasttext',
                            action='store_true',
                            help='use FastText embeddings to initialize lexical Model1 embeddings (dim. defined by FastText)')

        parser.add_argument(f'--{MODEL_PARAM_PREF}no_fasttext_embed_dim',
                            type=int,
                            metavar='embedding dim',
                            default=512,
                            help='Dimensionality of the lexical neural Model1')

        parser.add_argument(f'--{MODEL_PARAM_PREF}prob_self_tran', type=float, default=0.05,
                            metavar='self-train prob',
                            help='self-translation probability of the lexical neural Model1')

        parser.add_argument(f'--{MODEL_PARAM_PREF}proj_dim', type=int, default=128,
                            metavar='model1 projection dim',
                            help='neural lexical model1 projection dimensionionality')


