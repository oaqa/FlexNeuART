#!/usr/bin/env python
# grep*ForVal functions extract values of a given key
# from a string/file in a stupid one-entry-per-line format,
# where each line has the format (there can be spaces around :)
# key:value
import sys
file_name=sys.argv[1]
metr_name=sys.argv[2]
ignore_case=False
if len(sys.argv) >= 4:
    ignore_case=str(sys.argv[3]) == '1'

if ignore_case:
    metr_name = metr_name.lower()

with open(file_name) as f:
    for line in f:
        line = line.strip()
        sepi = line.find(':')
        if sepi >= 0:
            metr = line[0:sepi].strip()
            if ignore_case:
                metr = metr.lower()
            if metr == metr_name:
                print(line[sepi+1:].strip())
                sys.exit(0)

print('')
      
