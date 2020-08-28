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

questFile = open(sys.argv[1], 'r')
answFile = open(sys.argv[2], 'r')
maxFert = int(sys.argv[3])
outFileQuest = open(sys.argv[4], "w")
outFileAnsw = open(sys.argv[5], "w")
symmetr = int(sys.argv[6]) != 0
sampleProb = float(sys.argv[7])
assert sampleProb > 0
assert sampleProb <= 1 + 1e-6
print("Symmetrizing %d, max. fertility %d" % (symmetr, maxFert))

qty_orig = 0
qty_direct = 0
qty_flip = 0

random.seed(0) # always use the same zero seed for reproducibility

for quest in questFile:
    answ = answFile.readline()
    qty_orig += 1
    # We read the question and the answer,
    # here's time to make a random decsision:
    # to sample or not to sample

    if random.random() <= sampleProb:
        len1 = len(answ.split())
        len2 = len(quest.split())
        if len2 <= len1 * maxFert and len1 <= len2 * maxFert:
            outFileQuest.write(quest)
            outFileAnsw.write(answ)
            qty_direct += 1

            if symmetr:
                outFileQuest.write(answ)
                outFileAnsw.write(quest)
                qty_flip += 1

print(f'The sampling and filtering script processed {qty_orig} QA pairs and ' +
      f'wrote {qty_direct} original and {qty_flip} flipped pairs')

outFileQuest.close()
outFileAnsw.close()



