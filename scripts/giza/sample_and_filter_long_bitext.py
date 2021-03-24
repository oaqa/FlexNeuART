#!/usr/bin/env python
#
# Just a simple script to
# 1. sample bitext
# 2. ignore pairs where a question is much longer
#    then the respective document (or vice versa)
#
import sys
import random

if len(sys.argv) != 8:
    print("Usage: <questions> <answers> <maximum fertility> <output questions> <output answers> <symmetrize? 1 or 0> <sample prob>")

quest_file = open(sys.argv[1], 'r')
answ_file = open(sys.argv[2], 'r')
max_fert = int(sys.argv[3])
out_file_quest = open(sys.argv[4], "w")
out_file_answ = open(sys.argv[5], "w")
symmetr = int(sys.argv[6]) != 0
sample_prob = float(sys.argv[7])
assert sample_prob > 0
assert sample_prob <= 1 + 1e-6
print("Symmetrizing %d, max. fertility %d" % (symmetr, max_fert))

qty_orig = 0
qty_direct = 0
qty_flip = 0

random.seed(0) # always use the same zero seed for reproducibility

for quest in quest_file:
    answ = answ_file.readline()
    qty_orig += 1
    # We read the question and the answer,
    # here's time to make a random decsision:
    # to sample or not to sample

    if random.random() <= sample_prob:
        len1 = len(answ.split())
        len2 = len(quest.split())
        if len2 <= len1 * max_fert and len1 <= len2 * max_fert:
            out_file_quest.write(quest)
            out_file_answ.write(answ)
            qty_direct += 1

            if symmetr:
                out_file_quest.write(answ)
                out_file_answ.write(quest)
                qty_flip += 1

print(f'The sampling and filtering script processed {qty_orig} QA pairs and ' +
      f'wrote {qty_direct} original and {qty_flip} flipped pairs')

out_file_quest.close()
out_file_answ.close()



