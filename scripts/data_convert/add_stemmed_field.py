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
"""
    Adding a field stemmed using a Krovetz stemmer.
    Based on https://github.com/bmitra-msft/TREC-Deep-Learning-Quick-Start/blob/master/model_utils.py

    It requires installing an extra package: krovetzstemmer
"""
import argparse
import json

from tqdm import tqdm

from flexneuart.io import jsonl_gen, FileWrapper
from flexneuart.text_proc.parse import KrovetzStemParser
from flexneuart.io.stopwords import read_stop_words, STOPWORD_FILE

from flexneuart.config import TEXT_RAW_FIELD_NAME, TEXT_STEMMED_FIELD_NAME

parser = argparse.ArgumentParser(description='Add stemmed fields to the existing JSONL data entries')

parser.add_argument('--input', metavar='input JSONL file', help='input JSONL file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--output', metavar='output JSONL file', help='output JSONL file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--src_field', metavar='source field',
                    help='the name of field whose context is stemmed and stopped',
                    type=str, default=TEXT_RAW_FIELD_NAME)
parser.add_argument('--dst_field', metavar='target field',
                    help='the name of field to store the stemmed and stopped text',
                    type=str, default=TEXT_STEMMED_FIELD_NAME)


args = parser.parse_args()
print(args)
stop_words = read_stop_words(STOPWORD_FILE)
print(stop_words)

src_field = args.src_field
dst_field = args.dst_field

print(f'Source field: {src_field} target field: {dst_field}')


proc = KrovetzStemParser(stop_words=stop_words)

with FileWrapper(args.output, 'w') as outf:
    for doce in tqdm(jsonl_gen(args.input), desc='adding '):
        doce[dst_field] = proc(doce[src_field])

        outf.write(json.dumps(doce) + '\n')
