#!/usr/bin/env python
import sys
import numpy as np
import argparse
import tqdm

sys.path.append('.')

from scripts.data_convert.convert_common import FileWrapper
from scripts.data_convert.wikipedia_dpr.utils import dpr_json_reader, get_passage_id


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