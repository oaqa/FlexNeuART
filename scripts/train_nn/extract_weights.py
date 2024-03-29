#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Using some bits from CEDR: https://github.com/Georgetown-IR-Lab/cedr
#  which has MIT, i.e., Apache 2 compatible license.
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

"""
    A simple script to extract and save weights of previously trained model.
    This is useful when we want to train a model using a new set of parameters.
    If such parameters are saved with the model, they are not overriden by the
    configuration values.
"""

import argparse

from flexneuart.models.base import ModelSerializer

parser = argparse.ArgumentParser('model training and validation')

parser.add_argument('--input', metavar='input model file',
                    type=str, required=True, help='input model file previously generated by the framework')
parser.add_argument('--output', metavar='output file with weights',
                    type=str, required=True,
                    help='output file that will contain only model weights, but no framework parameters')

args = parser.parse_args()

model_holder : ModelSerializer = ModelSerializer.load_all(args.input)
model_holder.save_only_weights(args.output)
