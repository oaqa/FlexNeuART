#!/usr/bin/env python
# Computing top-K recall from NMSLIB-generated *.fnd files
import sys
import numpy as np
if len(sys.argv) != 4:
  sys.stderr.write('Usage: <input file> <desc file> <data file>\n')
  sys.exit(1)
data = np.genfromtxt(sys.argv[1], delimiter='\t')
with open(sys.argv[2], 'w') as descFile:
  with open(sys.argv[3], 'w') as dataFile:
    dataFile.write('label\ttop_k\trecall\n')
    n=1
    for r in data:
      label='id_%s' % n
      n=n+1
      descFile.write('%s\t%s\tmark=.,black\n' % (label, 'whatever'))
      for k in range(1, len(r)+1):
        val=np.mean(r[0:k])
        dataFile.write('%s\t%s\t%s\n' % (label, k, val))
