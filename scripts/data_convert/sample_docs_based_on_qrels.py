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
# A simple script that
# 1. reads QRELs and memorizes relevant documents
# 2. scans an input part, e.g., a bunch of passages
# 3. saves all the relevant ones
# 4. optionally save a sample of potentially non-relevant documents.
#
import sys
import os
import random
import json

sys.path.append('.')
import argparse
from scripts.eval_common import read_qrels_dict
from scripts.config import ANSWER_FILE_JSON, QREL_FILE, DOCID_FIELD
from scripts.data_convert.convert_common import jsonl_gen, FileWrapper

random.seed(0)

parser = argparse.ArgumentParser('Reading relevant and sampling non-relevant documents')

parser.add_argument('--qrel_dir',
                    metavar='qrel directory',
                    type=str,
                    help='a directory to read qrels from',
                    required=True)

parser.add_argument('--inp_doc_file',
                    metavar='input doc file',
                    type=str,
                    help=f'a file containing indexable document entries, e.g., {ANSWER_FILE_JSON}',
                    required=True)

parser.add_argument('--min_rel_grade',
                    metavar='min qrel relev. grade',
                    type=int,
                    required=True,
                    help='a minimum qrel grade, e.g., 2 to consider document relevant')

parser.add_argument('--out_doc_file',
                    metavar='out doc file',
                    type=str,
                    help=f'an output file with indexable document entries, e.g., {ANSWER_FILE_JSON}',
                    required=True)

parser.add_argument('--nonrel_sample_prob',
                    metavar='a prob. to sample non-relevant doc',
                    type=float,
                    help=f'a probability to sample non-relevant document entries',
                    required=True)


args = parser.parse_args()

sample_prob = args.nonrel_sample_prob

if sample_prob < 0 or sample_prob >= 1:
    print('Sampling probability must be >=0 and < 1')
    sys.exit(1)


qrel_dict = read_qrels_dict(os.path.join(args.qrel_dir, QREL_FILE))

all_rel_docs = set()

for qid, qd in qrel_dict.items():
    for did, rel in qd.items():
        if rel >= args.min_rel_grade:
            all_rel_docs.add(did)


with FileWrapper(args.out_doc_file, 'w') as out_file:
    for doc_entry in jsonl_gen(args.inp_doc_file):
        did = doc_entry[DOCID_FIELD]
        if did in all_rel_docs or random.random() < sample_prob:
            out_file.write(json.dumps(doc_entry) + '\n')

