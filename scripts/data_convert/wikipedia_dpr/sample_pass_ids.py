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
import numpy as np
import argparse
import tqdm

from flexneuart.io import FileWrapper
from flexneuart.data_convert.wikipedia_dpr import dpr_json_reader, get_passage_id


def parse_args():
    parser = argparse.ArgumentParser(description='Extract pasage IDs from all positive contexts and sample '
                                                 'passage IDs from negative ones')
    parser.add_argument('--input', metavar='input file list', help='input file list',
                        type=str, nargs='+', required=True)
    parser.add_argument('--output_pref', metavar='output ID file prefix',
                        help='a prefix of a file to store selected IDs',
                        type=str, required=True)

    args = parser.parse_args()

    return args


args = parse_args()
arg_vars=vars(args)

sel_psg_ids = set()

np.random.seed(0)

for inp_file in args.input:
    print(f'Processing {inp_file}')
    for fields in tqdm.tqdm(dpr_json_reader(FileWrapper(inp_file))):
        pos_ids = set()
        neg_ids = set()

        for entry in fields["positive_ctxs"]:
            pos_ids.add(get_passage_id(entry))

        for entry in fields["negative_ctxs"]:
            neg_ids.add(get_passage_id(entry))

        for entry in fields["hard_negative_ctxs"]:
            neg_ids.add(get_passage_id(entry))

        for psg_id in pos_ids:
            sel_psg_ids.add(psg_id)

        pos_qty = len(pos_ids)
        neg_qty = len(neg_ids)
        sel_qty = size=min(neg_qty, pos_qty)

        if sel_qty > 0:
            # Select min(neg_qty, pos_qty) negative context passage ids
            sel_neg_ids = np.random.choice(list(neg_ids), sel_qty, replace=False)
            for psg_id in sel_neg_ids:
                sel_psg_ids.add(psg_id)


print(f'Selected {len(sel_psg_ids)} passage ids')
np.save(file=args.output_pref, arr=np.array(list(sel_psg_ids)))
