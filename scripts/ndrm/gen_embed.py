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
    A script to compute fast text embeddings for a given (parsed) field for
    which we previously created a forward file with positions.
"""
import os
import fasttext
import argparse

from tqdm import tqdm

from flexneuart import configure_classpath
from flexneuart.retrieval import create_featextr_resource_manager
from flexneuart.retrieval.fwd_index import get_forward_index
from flexneuart.io import open_with_default_enc

parser = argparse.ArgumentParser(description='Generate field embeddings for NDRM models.')

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
                    metavar='output embedding file',
                    type=str,
                    required=True)
parser.add_argument('--num_hidden_nodes',
                    metavar='NDRM hidden size',
                    help='size of the NDRM hidden layer',
                    type=int, required=True)


args = parser.parse_args()

# add Java JAR to the class path
configure_classpath()

assert args.collect_dir is not None, "Specify a top-level collection directory!"
assert args.fwd_index_subdir is not None, "Specify the forward index directory!"
# create a resource manager
resource_manager = create_featextr_resource_manager(resource_root_dir=args.collect_dir,
                                                    fwd_index_dir=args.fwd_index_subdir)

fwd_index = get_forward_index(resource_manager, args.index_field)
fwd_index.check_is_parsed_text()

out_dir = os.path.dirname(args.output)
os.makedirs(out_dir, exist_ok=True)

temp_fn = os.path.join(out_dir, 'temp_file')

with open_with_default_enc(temp_fn, 'w') as fout:
    for doc_id in tqdm(fwd_index.get_all_doc_ids(), 'generating text from index'):
        fout.write(fwd_index.get_doc_entry_parsed_text(doc_id) + '\n')

# Defaults from NDRM code
embeddings = fasttext.train_unsupervised(temp_fn,
                                         model='skipgram',
                                         dim=args.num_hidden_nodes // 2,
                                         bucket=10000,
                                         minCount=100,
                                         minn=1,
                                         maxn=0,
                                         ws=10,
                                         epoch=5)
embeddings.save_model(args.output)

os.remove(temp_fn)