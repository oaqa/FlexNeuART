#!/usr/bin/env python
import argparse
import json
import os
import sys
import numpy as np

sys.path.append('scripts')

from config import *
from convert_common import readQueries

parser = argparse.ArgumentParser('Comparing external and internal eval tools')

parser.add_argument('--data_dir',
                    metavar='data directory',
                    help='data directory',
                    type=str, required=True)
parser.add_argument('--out_dir',
                    metavar='output directory',
                    help='output directory',
                    type=str, required=True)

args = parser.parse_args()
print(args)

out_dir = args.out_dir

if not os.path.exists(out_dir):
  os.makedirs(out_dir)

for subDir in os.listdir(args.data_dir):
  qf = os.path.join(args.data_dir, subDir, QUESTION_FILE_JSON)
  if os.path.exists(qf):
    print('Reading:', qf)
    res = []
    for e in readQueries(qf):
        res.append(e[DOCID_FIELD])

    print('Read', len(res), 'queries')
    np.save(os.path.join(out_dir, subDir + '.npy'),np.array(res))

