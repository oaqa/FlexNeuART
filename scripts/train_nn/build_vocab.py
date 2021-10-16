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
# Create a vocabulary from (a specific-field) contents of a JSONL input file field
#
import argparse

from tqdm import tqdm

from flexneuart.io.utils import jsonl_gen
from flexneuart.io.vocab import VocabBuilder

parser = argparse.ArgumentParser('Build vocabularies from several processed fiels')
parser.add_argument('--field_name', metavar='field name', help='a JSONL field to use', required=True)
parser.add_argument('--input', metavar='input files', help='input JSONL files (possibly compressed)',
                    type=str, nargs='+', required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)

args = parser.parse_args()
print(args)

vocab = VocabBuilder()
field = args.field_name

for fn in args.input:
    ln = 0
    for doc_entry in tqdm(jsonl_gen(fn), desc='Processing: ' + fn):
        ln += 1
        if field in doc_entry:
            vocab.proc_doc(doc_entry[field])
        else:
            print(f'WARNING: No field {field} is found in line {ln} file {fn}')
            continue

vocab.save(args.output)
