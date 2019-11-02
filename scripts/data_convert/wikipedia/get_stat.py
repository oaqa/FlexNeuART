#!/usr/bin/env python
import sys
sys.path.append('scripts')
from data_convert.convert_common import *

if len(sys.argv) != 2:
  print('Usage: <dump root dir>')
  sys.exit(1)


for fn in wikiExtractorFileIterator(sys.argv[1]):
  print(fn)
