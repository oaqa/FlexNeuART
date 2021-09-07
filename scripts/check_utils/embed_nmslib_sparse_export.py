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
# A script to convert sparse vectors export from NMSLIB into a "BSONL" format,
# (using target/appassembler/bin/ExportToNMSLIBSparse),
# which can be furhter processed by the creator of forward files.
# This scripts is exclusively for demonstration and debugging purposes, i.e.,
# to verify that the ranking components that compute the inner product between
# sparse vectors work properly.
#
import argparse
import struct

from tqdm import tqdm

from flexneuart.data_convert.utils import FileWrapper, DOCID_FIELD, ENDIANNES_TYPE, \
                                                pack_sparse_vect, write_json_to_bin, \
                                                read_and_unpack_int, read_ascii_str

parser = argparse.ArgumentParser(description='Store previously exported sparse vectors in "BSONL" format.')

parser.add_argument('--input', metavar='input file',
                    help='A previously exported query or document file (in the sparse NMSLIB format)',
                    type=str, required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)
parser.add_argument('--field_name', metavar='field name', help='the name of the BSONL field',
                    type=str, required=True)


args = parser.parse_args()
f = open(args.input, 'br')


with FileWrapper(args.output, 'wb') as out_file:

    rec_qty = read_and_unpack_int(f)

    for k in tqdm(range(rec_qty)):
        doc_id = read_ascii_str(f)
        dim = read_and_unpack_int(f)

        mask = f'{ENDIANNES_TYPE}' + ''.join(['If'] * dim)
        vect = struct.unpack(mask, f.read(8 * dim))

        data_elem = {DOCID_FIELD: doc_id, args.field_name: pack_sparse_vect(vect)}
        write_json_to_bin(data_elem, out_file)
