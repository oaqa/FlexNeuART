#!/usr/bin/env python
# A script to convert passages/documents or queries to dense vectors
import argparse
import os
import sys
import bson
import torch
import struct
import numpy as np
from tqdm import tqdm

sys.path.append('.')

from scripts.config import QUESTION_FILE_JSON
from scripts.data_convert.convert_common import FileWrapper, DOCID_FIELD, pack_dense_batch, write_json_to_bin
from scripts.data_convert.ance.ance_models import create_ance_firstp, create_dpr
from scripts.data_convert.ance.ance_data import DATA_TYPE_DPR_NQ, DATA_TYPE_DPR_TRIVIA, \
                                                DATA_TYPE_MSMARCO_DOC_FIRSTP, DATA_TYPE_MSMARCO_PASS, \
                                                DATA_TYPE_CHOICES, DATA_TYPE_PATHS, \
                                                msmarco_body_generator, wikipedia_dpr_body_generator, \
                                                jsonl_query_generator, tokenize_query_msmarco, tokenize_query_dpr


parser = argparse.ArgumentParser(description='Convert passages and/or documents to dense vectors.')

parser.add_argument('--input', metavar='input file',
                    help='input file (query JSONL or raw passage/document intput file)',
                    type=str, required=True)
parser.add_argument('--batch_size', metavar='batch size', help='batch size',
                    type=int, default=16)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)
parser.add_argument('--field_name', metavar='field name', help='the name of the BSONL field',
                    type=str, required=True)
parser.add_argument('--model_dir', metavar='model download directory',
                    help='a directory with downloaded and unpacked models',
                    type=str, required=True)
parser.add_argument('--data_type', metavar='data type',
                    help='data type: ' + ', '.join(DATA_TYPE_CHOICES),
                    choices=DATA_TYPE_CHOICES,
                    required=True)
parser.add_argument('--doc_ids', metavar='optional document/passage ids',
                    type=str, default=None, help='an optional numpy array with passage ids to select')

args = parser.parse_args()

flt_doc_ids = None
if args.doc_ids is not None:
    flt_doc_ids = set(np.load(args.doc_ids))
    print(f'Restricting parsing to {len(flt_doc_ids)} document/passage IDs')

data_type = args.data_type

assert data_type in DATA_TYPE_PATHS

model_path = os.path.join(args.model_dir, DATA_TYPE_PATHS[data_type])
print('Model path', model_path)

is_query =

print(f'Query?: {is_query}')

if data_type in [DATA_TYPE_MSMARCO_DOC_FIRSTP, DATA_TYPE_MSMARCO_PASS]:
    is_doc = data_type == DATA_TYPE_MSMARCO_DOC_FIRSTP

    print('Creating ANCE FirstP type model')
    model, tokenizer = create_ance_firstp(model_path)
    if is_query:
        text_generator = jsonl_query_generator(args.input, tokenizer, tokenize_query_msmarco)
    else:
        text_generator = msmarco_body_generator(args.input, is_doc, tokenizer)

elif data_type in [DATA_TYPE_DPR_NQ, DATA_TYPE_DPR_TRIVIA]:

    print('Creating DPR type model')
    model, tokenizer = create_dpr(model_path)
    if is_query:
        text_generator = jsonl_query_generator(args.input, tokenizer, tokenize_query_dpr)
    else:
        text_generator = wikipedia_dpr_body_generator(args.input, tokenizer)

else:
    # Shouldn't happen unless the code is changed to add more data choices
    raise Exception(f'Unsupported data type: {data_type}')


print('Text generator:', text_generator)

model.cuda()

batch_input = []

def proc_batch(batch_input, is_query, model, out_file, field_name):
    if batch_input:
        doc_ids, tok_ids, attn_masks = zip(*batch_input)

        bqty = len(batch_input)

        attn_masks = torch.LongTensor(attn_masks).cuda()
        tok_ids = torch.LongTensor(tok_ids).cuda()

        if is_query:
            batch_data = model.query_emb(tok_ids, attn_masks)
        else:
            batch_data = model.body_emb(tok_ids, attn_masks)

        #print(tok_ids.shape, '!!', attn_masks.shape, '##', torch.sum(attn_masks, dim=-1), '@@', batch_data.shape)

        batch_data_packed = pack_dense_batch(batch_data)

        assert len(batch_data_packed) == bqty

        for i in range(bqty):
            data_elem = {DOCID_FIELD : doc_ids[i], field_name : batch_data_packed[i]}
            write_json_to_bin(data_elem, out_file)

        batch_input.clear()


with FileWrapper(args.output, 'wb') as out_file:
    for doc_id, attn_masks, tok_ids in tqdm(text_generator):
        if flt_doc_ids is not None:
            if doc_id not in flt_doc_ids:
                continue

        batch_input.append((doc_id, attn_masks, tok_ids))

        if len(batch_input) >= args.batch_size:
            proc_batch(batch_input, is_query, model, out_file, args.field_name)


proc_batch(batch_input, is_query, model, out_file, args.field_name)

