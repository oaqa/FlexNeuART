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
# A simple script to print model 1 associations
# assumes that the number of GIZA iterations was 5
#
import sys
import os

from tqdm import tqdm

if len(sys.argv) != 4:
    print('Usage <Orig. GIZA directory (not compressed & filtered one)> <source word id> <min. prob>') 
    sys.exit(1)

giza_dir = sys.argv[1]
src_vid = int(sys.argv[2])
min_prob = float(sys.argv[3])

src_voc = os.path.join(giza_dir, 'source.vcb')
trg_voc = os.path.join(giza_dir, 'target.vcb')
tran_file = os.path.join(giza_dir, 'output.t1.5')

def read_vocab(fn):
    res = {}
    with open(fn) as f:
        for line in tqdm(f, 'reading '+fn):
            src_id, word, _ = line.split()
            res[src_id] = word
    return res

src_voc = read_vocab(src_voc)
trg_voc = read_vocab(trg_voc)


with open(tran_file) as f:
    for line in f:
        vid1, vid2, p = line.split()
        p = float(p)
        if int(vid1) == src_vid and p >= min_prob: 
            print(src_voc[vid1], trg_voc[vid2], p)
