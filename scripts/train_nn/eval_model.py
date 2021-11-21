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
    A script to evaluate a single model using a given RUN file.
"""
import os
import argparse
from tqdm import tqdm

from flexneuart.config import DEFAULT_VAL_BATCH_SIZE
from flexneuart.models.train import run_model
from flexneuart.models.utils import add_model_init_basic_args
from flexneuart.models.base import ModelSerializer

from flexneuart.io.queries import read_queries_dict
from flexneuart.io.json import save_json
from flexneuart.io.runs import read_run_dict

from flexneuart import configure_classpath
from flexneuart.retrieval import create_featextr_resource_manager
from flexneuart.retrieval.fwd_index import get_forward_index

from flexneuart.eval import METRIC_LIST, get_eval_results

from time import time

parser = argparse.ArgumentParser(description='Run basic run checks')
parser.add_argument('--run_orig', metavar='original run file',
                    help='an original run file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--run_rerank', metavar='a re-ranked run file',
                    help='an optional re-ranked run file (can be compressed)',
                    type=str, required=True)
parser.add_argument('--query_file', metavar='query file',
                    help='a query file',
                    type=str, required=True)
parser.add_argument('--summary_json', metavar='validation summary JSON file',
                    help='a file with validation summary statistics',
                    type=str, required=True)
parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                    type=str, required=True)
parser.add_argument('--collect_dir', metavar='collection dir',
                    help='a top-level collection directory',
                    required=True)
parser.add_argument('--fwd_index_subdir',
                    help='forward index files sub-directory',
                    required=True)
parser.add_argument('--index_field', metavar='index field',
                    help='the name of the field for which we previously created the forward index',
                    type=str,
                    required=True)
parser.add_argument('--query_field', metavar='query field',
                    help='the name of the query field: defaults to the index field name',
                    type=str,
                    default=None)
parser.add_argument('--cand_score_weight', metavar='candidate provider score weight',
                    type=float, default=0.0,
                    help='a weight of the candidate generator score used to combine it with the model score.')
parser.add_argument('--keep_case', action='store_true',
                    help='no lower-casing')
parser.add_argument('--batch_size', metavar='batch size',
                    default=DEFAULT_VAL_BATCH_SIZE, type=int,
                    help='batch size')
parser.add_argument('--max_num_query', metavar='max # of val queries',
                    type=int, default=None,
                    help='max # of validation queries (for debug purposes only)')
parser.add_argument('--eval_metric', choices=METRIC_LIST, default=METRIC_LIST[0],
                    help='Metric list: ' + ','.join(METRIC_LIST),
                    metavar='eval metric')

add_model_init_basic_args(parser,
                          add_device_name=True,
                          add_init_model_weights=False, mult_model=False)

args = parser.parse_args()

# add Java JAR to the class path
configure_classpath()

assert args.collect_dir is not None, "Specify a top-level collection directory!"
assert args.fwd_index_subdir is not None, "Specify the forward index directory!"
# create a resource manager
resource_manager=create_featextr_resource_manager(resource_root_dir=args.collect_dir,
                                              fwd_index_dir=args.fwd_index_subdir)

fwd_index = get_forward_index(resource_manager, args.index_field)

fwd_index.check_is_text_raw()

fname = args.init_model.name
print('Loading model from:', fname)
model_holder = ModelSerializer.load_all(fname)
max_doc_len = model_holder.max_doc_len
max_query_len = model_holder.max_query_len

query_field = args.query_field
if query_field is None:
    query_field = args.index_field

device_name=args.device_name
model = model_holder.model
model.to(device_name)

max_query_val = args.max_num_query

print(f'Device: {device_name}')
print(f'max # of queries: {max_query_val} max query/document lengths: {max_query_len}/{max_doc_len}, keep case? {args.keep_case}')
print(f'(Index field: {args.index_field} query field: {query_field}')

do_lower_case = not args.keep_case

query_dict = {}
for qid, e in read_queries_dict(args.query_file).items():
    query_text = e[query_field]

    if do_lower_case:
        query_text = query_text.lower()
    query_dict[qid] = query_text

data_dict = {}

dataset = query_dict, data_dict

orig_run = read_run_dict(args.run_orig)

query_ids = list(orig_run.keys())
if max_query_val is not None:
    query_ids = query_ids[0:max_query_val]
    valid_run = {k: orig_run[k] for k in query_ids}
else:
    valid_run = orig_run

for qid, query_scores in tqdm(valid_run.items(), 'reading documents'):
    for did, _ in query_scores.items():
        if did not in data_dict:
            doc_text = fwd_index.get_doc_text_raw(did)
            if do_lower_case:
                doc_text = doc_text.lower()

            data_dict[did] = doc_text

start_val_time = time()
rerank_run = run_model(model,
              device_name=device_name,
              batch_size=args.batch_size, amp=args.amp,
              max_query_len=max_query_len, max_doc_len=max_doc_len,
              dataset=dataset, orig_run=valid_run,
              cand_score_weight=args.cand_score_weight,
              desc='validating the run')
end_val_time = time()

# HF tokenizers do not "like" to be forked, but it doesn't matter at this point
# So we just use this variable to disable the warning
# see, e.g., https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
metric_name=args.eval_metric.lower()
query_qty=len(rerank_run)
valid_score = get_eval_results(use_external_eval=True, # Must use trec_eval here, which is an official eval tool
                          eval_metric=metric_name,
                          rerank_run=rerank_run,
                          qrel_file=args.qrels,
                          run_file=args.run_rerank)

print(f'Metric {metric_name} score: {valid_score} # of queries: {query_qty}')

valid_stat= {
     'score': valid_score,
     'validation_time': end_val_time - start_val_time
}

save_json(args.summary_json, valid_stat)


