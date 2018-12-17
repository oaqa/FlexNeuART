#!/usr/bin/env python
import subprocess as sp
seen_core=set()
output=sp.check_output(["cat", "/proc/cpuinfo"]).replace('\t', ' ').split('\n')
start=0
output.append('') # There would always a trailing empty line
numCores=0
while (start < len(output)):
  line=output[start].strip()
  if line != '':
    end = start +1
    while (end < len(output)):
      if output[end].strip() == '' : break
      end=end+1
    #print start, end
    id=0
    nonRepeat=False
    for i in range(start, end):
      line=output[i].strip()
      fields=line.split(':')
      if (len(fields)==2 and fields[0].strip() == 'physical id'):
        id=fields[1].strip()
        if not id in seen_core:
          seen_core.add(id)
          nonRepeat=True
          break
    if nonRepeat:
      for i in range(start, end):
        line=output[i].strip()
        fields=line.split(':')
        if (len(fields)==2 and fields[0].strip() == 'siblings'):
          numCores += int(fields[1].strip())
    start = end + 1
  start = start + 1

print(numCores)
