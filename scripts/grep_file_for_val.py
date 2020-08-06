#!/usr/bin/env python
# grep*ForVal functions extract values of a given key
# from a string/file in a stupid one-entry-per-line format,
# where each line has the format (there can be spaces around :)
# key:value
import sys
fileName=sys.argv[1]
metrName=sys.argv[2]
ignoreCase=False
if len(sys.argv) >= 4:
    ignoreCase=str(sys.argv[3]) == '1'

if ignoreCase:
    metrName = metrName.lower()

with open(fileName) as f:
    for line in f:
        line = line.strip()
        sepi = line.find(':')
        if sepi >= 0:
            metr = line[0:sepi].strip()
            if ignoreCase:
                metr = metr.lower()
            if metr == metrName:
                print(line[sepi+1:].strip())
                sys.exit(0)

print('')
      
