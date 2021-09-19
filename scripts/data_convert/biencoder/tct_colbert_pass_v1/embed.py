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
    A script to convert MS MARCO v1 passages or queries to dense vectors using PySerini's TCT-COLBERT models.
    These vectors are stored in "BSONL" format, which can be used to create a forward index.
"""
import argparse
import numpy as np
from tqdm import tqdm

from flexneuart.config import DEFAULT_DEVICE_GPU

DEVICE_NAME=DEFAULT_DEVICE_GPU

from flexneuart.config import  DOCID_FIELD, TEXT_RAW_FIELD_NAME
from flexneuart.io import FileWrapper, jsonl_gen
from flexneuart.io.queries import is_json_query_file
from flexneuart.io.pack import pack_dense_batch, write_json_to_bin

from flexneuart.data_convert.biencoder.tct_colbert_pass_v1.models import TctColBertDocumentEncoder, TctColBertQueryEncoder


parser = argparse.ArgumentParser(description='Convert passages and/or documents to dense vectors and store them in "BSONL" format.')

parser.add_argument('--input', metavar='input file',
                    help='input file (query JSONL or raw passage/document input file)',
                    type=str, required=True)
parser.add_argument('--batch_size', metavar='batch size', help='batch size',
                    type=int, default=16)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)
parser.add_argument('--field_name', metavar='field name', help='the name of the BSONL field',
                    type=str, required=True)
parser.add_argument('--max_len', metavar='maximum # of tokens', help='the maximum input length (in the # of subword tokens)',
                    type=int, default=512)
parser.add_argument('--model', metavar='model name or path',
                    help='a directory with downloaded and unpacked models',
                    type=str, default="castorini/tct_colbert-msmarco")
parser.add_argument('--amp', action='store_true', help="Use automatic mixed-precision")

args = parser.parse_args()

print('Model', args.model)

is_query = is_json_query_file(args.input)

print(f'Query?: {is_query}')

if is_query:
    model = TctColBertQueryEncoder(args.model, device=DEVICE_NAME)
else:
    model = TctColBertDocumentEncoder(args.model, amp=args.amp, device=DEVICE_NAME)


batch_input = []

def proc_batch(batch_input, is_query, model, out_file, field_name):
    if batch_input:
        doc_ids, texts = zip(*batch_input)

        bqty = len(batch_input)

        if is_query:
            # It's unfortunate to encode them one-by-one, but we try
            # to preserve the original PySerini encoder (which is not batched)
            # b/c query encoding i supposed to happen on CPU
            batch_data_arr = []
            for query in texts:
                batch_data_arr.append(model.encode(query))
            batch_data = np.vstack(batch_data_arr)
        else:
            batch_data = model.encode(texts, max_length=args.max_len)
        #print(batch_data.shape)

        batch_data_packed = pack_dense_batch(batch_data)

        assert len(batch_data_packed) == bqty

        for i in range(bqty):
            data_elem = {DOCID_FIELD : doc_ids[i], field_name : batch_data_packed[i]}
            write_json_to_bin(data_elem, out_file)

        batch_input.clear()


with FileWrapper(args.output, 'wb') as out_file:
    for e in tqdm(jsonl_gen(args.input)):
        doc_id = e[DOCID_FIELD]
        text = e[TEXT_RAW_FIELD_NAME]

        batch_input.append((doc_id, text))

        if len(batch_input) >= args.batch_size:
            proc_batch(batch_input, is_query, model, out_file, args.field_name)

    proc_batch(batch_input, is_query, model, out_file, args.field_name)
