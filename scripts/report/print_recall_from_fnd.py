#!/usr/bin/env python
# Computing top-K recall from NMSLIB-generated *.fnd files
import sys
import numpy as np
if len(sys.argv) != 3:
  sys.stderr.write('Usage: <input file> <top-K>\n')
  sys.exit(1)
data = np.genfromtxt(sys.argv[1], delimiter='\t')
K=int(sys.argv[2])
for r in data:
  print np.mean(r[0:K])
