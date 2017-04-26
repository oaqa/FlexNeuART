#!/usr/bin/env python
# This script is supposed to read the index file from stdin

import sys
import re

dataQty=0
linkQty=0.0
for line in sys.stdin:
  fields=line.strip().split()
  if len(fields) > 1:
    f=fields[0]
    # The line that specifies data point neighbors
    if f.endswith(':') and len(f.split(':')) == 3 and f.split(':')[2] == '':
      dataQty = dataQty + 1
      linkQty = linkQty + len(fields) - 1


print "The number of data points %d" % dataQty
print "Average number of neighbors %f" % (linkQty/dataQty)
      
