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
import sys
import argparse
import torch

from flexneuart.models.train.amp import get_amp_processors
from flexneuart.models.utils import add_model_init_basic_args
from flexneuart.models.base import ModelSerializer

from flexneuart.text_proc import handle_case

from flexneuart.featextr_server.python_generated.protocol.ExternalScorer import TextEntryRaw
from flexneuart.featextr_server.base import BaseQueryHandler, start_query_server


from flexneuart.models.train.batch_obj import BatchObject
from flexneuart.models.train.batching import BatchingValidationGroupByQuery

DEFAULT_BATCH_SIZE = 32

class RankQueryHandler(BaseQueryHandler):
    # Exclusive==True means single-threaded processing, which seems to be necessary here (there were hang ups otherwise)
    def __init__(self,
                    model_list,
                    keep_case,
                    batch_size, device_name,
                    max_query_len, max_doc_len,
                    cand_score_weight,
                    exclusive,
                    amp,
                    debug_print=False):
        super().__init__(exclusive=exclusive)

        self.debug_print = debug_print
        self.batch_size = batch_size

        self.amp = amp
        self.do_lower_case = not keep_case

        self.cand_score_weight = cand_score_weight

        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.device_name = device_name
        print('Maximum query/document len %d/%d device: %s' % (self.max_query_len, self.max_doc_len, self.device_name))

        self.model_list = model_list
        for model in self.model_list:
            model.to(self.device_name)
            # need to be in the eval mode
            model.eval()

    def handle_case(self, text: str):
        return handle_case(self.do_lower_case, text)

    def compute_scores_from_parsed_override(self, query, docs):
        # Sending words array with subsequent concatenation is quite inefficient in Python (very-very inefficient)
        # This is left for compatibility, but you better use the option sendParsedAsRaw in the JSON config file
        # for a Java-based feature extractor
        query_raw = TextEntryRaw(query.id, self.concat_text_entry_words(query))
        docs_raw = []
        for e in docs:
            docs_raw.append(TextEntryRaw(e.id, self.concat_text_entry_words(e)))

        return self.compute_scores_from_raw_override(query_raw, docs_raw)

    def compute_scores_from_raw_override(self, query, docs):
        print('Processing query:', query.id, query.text, '# of docs: ', len(docs))

        query_data = {query.id: self.handle_case(query.text)}
        # Run maps queries to arrays of document IDs see iter_valid_records (train_model.py)
        run = {query.id: {e.id : 0 for e in docs}}

        doc_data = {}
        for e in docs:
            doc_data[e.id] = self.handle_case(e.text)

        model_qty = len(self.model_list)
        # Initialize the return dictionary: for each document ID, a zero-element array of the size # of models.
        sample_ret = {e.id : [0.] * model_qty for e in docs}

        auto_cast_class, _ = get_amp_processors(self.amp)

        if doc_data:

            # based on the code from run_model function (train_model.py)
            data_set = query_data, doc_data
            # must disable gradient computation to greatly reduce memory requirements and speed up things
            with torch.no_grad():
                for model_id, model in enumerate(self.model_list):
                    iter_val = BatchingValidationGroupByQuery(batch_size=self.batch_size,
                                                              dataset=data_set, model=self.model,
                                                              max_query_len=self.max_query_len,
                                                              max_doc_len=self.max_doc_len,
                                                              run=run)

                    for batch in iter_val():
                        batch: BatchObject = batch
                        batch.to(self.device_name)
                        model_scores = self.model(*batch.features)
                        assert len(model_scores) == len(batch)
                        scores = model_scores + batch.cand_scores * self.cand_score_weight
                        # tolist() works much faster compared to extracting scores one by one using .item()
                        scores = scores.tolist()

                        for qid, did, score in zip(batch.query_ids, batch.doc_ids, scores):
                            if self.debug_print:
                                print('model id:', model_id, 'score & doc. id:', score, did, doc_data[did])

                            assert did in sample_ret, f'Bug: missing document ID: {did} in the result set.'
                            sample_ret[did][model_id] = score

        if self.debug_print:
            print('All scores:', sample_ret)

        return sample_ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A ranking server.')

    add_model_init_basic_args(parser, add_device_name=True, add_init_model_weights=False, mult_model=True)

    parser.add_argument('--debug_print', action='store_true',
                        help='Provide debug output')

    parser.add_argument('--batch_size', metavar='batch size',
                        default=DEFAULT_BATCH_SIZE, type=int,
                        help='batch size')

    parser.add_argument('--port', metavar='server port',
                        required=True, type=int,
                        help='Server port')

    parser.add_argument('--host', metavar='server host',
                        default='127.0.0.1', type=str,
                        help='server host addr to bind the port')

    parser.add_argument('--keep_case', action='store_true',
                        help='no lower-casing')
 
    parser.add_argument('--cand_score_weight', metavar='candidate provider score weight',
                        type=float, default=0.0,
                        help='a weight of the candidate generator score used to combine it with the model score.')

    parser.add_argument('--amp', action='store_true',
                        help="Use automatic mixed-precision")


    args = parser.parse_args()

    print(args)

    model_list = []

    assert(args.init_model_list is not None and type(args.init_model_list) == list)

    model_list = []

    all_max_query_len = None
    all_max_doc_len = None

    if args.init_model_list is not None:
        for model_file in args.init_model_list:
            fname = model_file.name

            print('Loading model from:', fname)

            model_holder = ModelSerializer.load_all(fname)

            if all_max_doc_len is None:
                all_max_doc_len = model_holder.max_doc_len
                all_max_query_len = model_holder.max_query_len
            else:
                if all_max_doc_len != model_holder.max_doc_len:
                    print(f'Inconsistent max. doc len, previous value {all_max_doc_len} '
                          f'model {fname} has: {model_holder.max_doc_len}')
                    sys.exit(1)
                if all_max_query_len != model_holder.max_query_len:
                    print(f'Inconsistent max. query len, previous value {all_max_query_len} '
                          f'model {fname} has: {model_holder.max_query_len}')
                    sys.exit(1)

            model_list.append(model_holder.model)
    else:
        print('Specify model files: --init_model_list')
        sys.exit(1)


    print(f'Max query/document lengths: {all_max_query_len}/{all_max_doc_len}')


    multi_threaded = False  # if we set to True, we can often run out of CUDA memory.
    start_query_server(args.host, args.port, multi_threaded, RankQueryHandler(model_list=model_list,
                                                                              amp=args.amp,
                                                                              keep_case=args.keep_case,
                                                                              batch_size=args.batch_size,
                                                                              debug_print=args.debug_print,
                                                                              device_name=args.device_name,
                                                                              max_query_len=all_max_query_len,
                                                                              max_doc_len=all_max_doc_len,
                                                                              cand_score_weight=args.cand_score_weight,
                                                                              exclusive=not multi_threaded))
