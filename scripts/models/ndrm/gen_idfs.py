#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Reusing bits of NDRM code, which is distrubted using Apache2-compabitlbe
#  MIT-license:
#  https://github.com/bmitra-msft/TREC-Deep-Learning-Quick-Start
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
    A script to compute and store IDFs of terms for a given (parsed) field for
    which we previously created a parsed index (with or without positions).
"""
import os
import math
import argparse

from tqdm import tqdm

from flexneuart import configure_classpath
from flexneuart.retrieval import create_featextr_resource_manager
from flexneuart.retrieval.fwd_index import get_forward_index
from flexneuart.io import open_with_default_enc

parser = argparse.ArgumentParser(description='Generate IDFs for NDRM models.')

parser.add_argument('--collect_dir', metavar='collection dir',
                    help='a top-level collection directory',
                    default=None)
parser.add_argument('--fwd_index_subdir',
                    help='forward index files sub-directory',
                    default=None)
parser.add_argument('--index_field', metavar='index field',
                    help='the name of the field for which we previously created the forward index',
                    type=str,
                    default=None)
parser.add_argument('--output',
                    metavar='output IDF file',
                    type=str,
                    required=True)


args = parser.parse_args()

# add Java JAR to the class path
configure_classpath()

assert args.collect_dir is not None, "Specify a top-level collection directory!"
assert args.fwd_index_subdir is not None, "Specify the forward index directory!"
# create a resource manager
resource_manager = create_featextr_resource_manager(resource_root_dir=args.collect_dir,
                                                    fwd_index_dir=args.fwd_index_subdir)

fwd_index = get_forward_index(resource_manager, args.index_field)
fwd_index.check_is_parsed()

doc_qty = fwd_index.get_doc_qty()

idfs = {}
for word_id in tqdm(fwd_index.get_all_word_ids(), 'computing IDFs'):
    we = fwd_index.get_word_entry_by_id(word_id)
    word = fwd_index.get_word_by_id(word_id)
    v = we.word_freq
    idfs[word] = max(math.log((doc_qty - v + 0.5) / (v + 0.5)), 0)

idfs = {k: v for k, v in idfs.items() if v > 0}
idfs = sorted(idfs.items(), key=lambda kv: kv[1])

out_dir = os.path.dirname(args.output)
os.makedirs(out_dir, exist_ok=True)

with open_with_default_enc(args.output, 'w') as f:
    for (k, v) in idfs:
        f.write('{}\t{}\n'.format(k, v))






