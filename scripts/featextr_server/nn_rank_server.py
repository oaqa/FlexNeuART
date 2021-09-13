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

from flexneuart.models.utils import add_model_init_basic_args
from flexneuart.models.base import ModelSerializer

from flexneuart.featextr_server.python_generated.protocol.ExternalScorer import TextEntryRaw
from flexneuart.featextr_server.base import BaseQueryHandler, start_query_server

import flexneuart.models.train.data as data

from flexneuart.models.train.data import DOC_TOK_FIELD, DOC_MASK_FIELD, \
    QUERY_TOK_FIELD, QUERY_MASK_FIELD, QUERY_ID_FIELD, DOC_ID_FIELD

DEFAULT_BATCH_SIZE = 32

class RankQueryHandler(BaseQueryHandler):
    # Exclusive==True means single-threaded processing, which seems to be necessary here (there were hang ups otherwise)
    def __init__(self,
                    model_list,
                    batch_size, device_name,
                    max_query_len, max_doc_len,
                    exclusive,
                    debug_print=False):
        super().__init__(exclusive=exclusive)

        self.debug_print = debug_print
        self.batch_size = batch_size

        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.device_name = device_name
        print('Maximum query/document len %d/%d device: %s' % (self.max_query_len, self.max_doc_len, self.device_name))

        self.model_list = model_list
        for model in self.model_list:
            model.to(self.device_name)
            # need to be in the eval mode
            model.eval()


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

        query_data = {query.id: query.text}
        # Run maps queries to arrays of document IDs see iter_valid_records (train_model.py)
        run = {query.id: {e.id : 0 for e in docs}}

        doc_data = {}
        for e in docs:
            doc_data[e.id] = e.text

        model_qty = len(self.model_list)
        # Initialize the return dictionary: for each document ID, a zero-element array of the size # of models.
        sample_ret = {e.id : [0.] * model_qty for e in docs}

        if doc_data:

            # based on the code from run_model function (train_model.py)
            data_set = query_data, doc_data
            # must disable gradient computation to greatly reduce memory requirements and speed up things
            with torch.no_grad():
                for model_id, model in enumerate(self.model_list):
                    for records in data.iter_valid_records(model, self.device_name, data_set, run,
                                                           self.batch_size,
                                                           self.max_query_len, self.max_doc_len):

                        scores = model(records[QUERY_TOK_FIELD],
                                       records[QUERY_MASK_FIELD],
                                       records[DOC_TOK_FIELD],
                                       records[DOC_MASK_FIELD])


                        # tolist() works much faster compared to extracting scores
                        # one by one using .item()
                        scores = scores.tolist()

                        for qid, did, score in zip(records[QUERY_ID_FIELD], records[DOC_ID_FIELD], scores):
                            if self.debug_print:
                                print('model id:', model_id, 'score & doc. id:', score, did, doc_data[did])

                            assert did in sample_ret, f'Bug: missing document ID: {did} in the result set.'
                            sample_ret[did][model_id] = score

        if self.debug_print:
            print('All scores:', sample_ret)

        return sample_ret



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serving CEDR models.')

    add_model_init_basic_args(parser, add_device_name=True, add_init_model_weights=False)

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


    args = parser.parse_args()

    print(args)

    model_list = []

    assert(args.init_model is not None and type(args.init_model) == list)

    model_list = []

    all_max_query_len = None
    all_max_doc_len = None

    if args.init_model is not None:
        for fname in args.init_model.name:
            model_holder: ModelSerializer = ModelSerializer(args.model)

            print('Loading model weights from:', fname)

            model_holder.load_all(fname)

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
        print('Specify the model file: --init_model')
        sys.exit(1)


    print(f'Max query/document lenghts: {all_max_query_len}/{all_max_doc_len}')


    multi_threaded = False  # if we set to True, we can often run out of CUDA memory.
    start_query_server(args.host, args.port, multi_threaded, RankQueryHandler(model_list=model_list,
                                                                              batch_size=args.batch_size,
                                                                              debug_print=args.debug_print,
                                                                              device_name=args.device_name,
                                                                              max_query_len=all_max_query_len,
                                                                              max_doc_len=all_max_doc_len,
                                                                              exclusive=not multi_threaded))
