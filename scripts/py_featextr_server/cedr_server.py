#!/usr/bin/env python
import sys
import argparse
import torch
import time

sys.path.append('.')

from scripts.py_featextr_server.python_generated.protocol.ttypes import TextEntryRaw
from scripts.py_featextr_server.base_server import BaseQueryHandler, start_query_server

import scripts.cedr.model_init_utils as model_init_utils
import scripts.cedr.data as data

DEFAULT_BATCH_SIZE = 32

class CedrQueryHandler(BaseQueryHandler):
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
        # Run maps queries to arrays of document IDs see iter_valid_records (train.py)
        run = {query.id: [e.id for e in docs]}

        doc_data = {}
        for e in docs:
            doc_data[e.id] = e.text

        model_qty = len(self.model_list)
        # Initialize the return dictionary: for each document ID, a zero-element array of the size # of models.
        sample_ret = {e.id : [0.] * model_qty for e in docs}

        if doc_data:

            # based on the code from run_model function (train.py)
            data_set = query_data, doc_data
            # must disable gradient computation to greatly reduce memory requirements and speed up things
            with torch.no_grad():
                for model_id, model in enumerate(self.model_list):
                    for records in data.iter_valid_records(model, self.device_name, data_set, run,
                                                           self.batch_size,
                                                           self.max_query_len, self.max_doc_len):

                        scores = model(records['query_tok'],
                                        records['query_mask'],
                                        records['doc_tok'],
                                        records['doc_mask'])


                        # tolist() works much faster compared to extracting scores
                        # one by one using .item()
                        scores = scores.tolist()

                        for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                            if self.debug_print:
                                print('model id:', model_id, 'score & doc. id:', score, did, doc_data[did])

                            assert did in sample_ret, f'Bug: missing document ID: {did} in the result set.'
                            sample_ret[did][model_id] = score

        if self.debug_print:
            print('All scores:', sample_ret)

        return sample_ret


def add_eval_model_init_args(parser):

    parser.add_argument('--model', metavar='model',
                        help='a model to use: ' + ' '.join(list(model_init_utils.MODEL_MAP.keys())),
                        choices=model_init_utils.MODEL_MAP.keys(), default='vanilla_bert')

    parser.add_argument('--init_model_weights',
                        metavar='model weights', help='initial model weights',
                        type=argparse.FileType('rb'),
                        nargs='+',
                        default=None)

    parser.add_argument('--init_model',
                        metavar='initial model',
                        help='initial *COMPLETE* model with heads and extra parameters',
                        type=argparse.FileType('rb'),
                        nargs='+',
                        default=None)

    parser.add_argument('--max_query_len', metavar='max. query length',
                        type=int, default=data.DEFAULT_MAX_QUERY_LEN,
                        help='max. query length')

    parser.add_argument('--max_doc_len', metavar='max. document length',
                        type=int, default=data.DEFAULT_MAX_DOC_LEN,
                        help='max. document length')


    parser.add_argument('--device_name', metavar='CUDA device name or cpu', default='cuda:0',
                        help='The name of the CUDA device to use')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serving CEDR models.')

    add_eval_model_init_args(parser)

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

    model_list = []

    assert(args.init_model is None or type(args.init_model) == list)
    assert (args.init_model_weights is None or type(args.init_model_weights) == list)

    model_list = []

    if args.init_model is None:
        # TODO this one isn't properly tested
        if args.model is not None and args.init_model_weights is not None:
            for fname in args.init_model_weights.name:
                model = model_init_utils.create_model_from_args(args)
                print('Loading model weights from:', fname.name)
                # If we load weights here, we must set strict to True:
                # this would prevent accidental loading of partial models.
                # Partial models are sure fine to load during training (as a reasonable
                # initialization), but not during test time.
                model.load_state_dict(torch.load(fname.name, map_location='cpu'),
                                      strict=True)
                model_list.append(model)
        else:
            print('Specify the model file: --init_model or model type and model weights')
            sys.exit(1)
    else:
        for fname in args.init_model:
            print('Loading model from:', fname.name)
            model = torch.load(fname.name, map_location='cpu')
            model_list.append(model)

    multi_threaded = False  # if we set to True, we can often run out of CUDA memory.
    start_query_server(args.host, args.port, multi_threaded, CedrQueryHandler(model_list=model_list,
                                                                           batch_size=args.batch_size,
                                                                           debug_print=args.debug_print,
                                                                           device_name=args.device_name,
                                                                           max_query_len=args.max_query_len,
                                                                           max_doc_len=args.max_doc_len,
                                                                           exclusive=not multi_threaded))
