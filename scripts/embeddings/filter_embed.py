#!/usr/bin/env python

import sys
import random

f=open(sys.argv[1])
maxWordQty=int(sys.argv[2])

if maxWordQty <= 0: raise Exception("maxWordQty should be a positive integer!")

f.readline()
s=f.readline().rstrip()
if s != '': raise Exception("We expect the second line to be empty!")

dictList = []
for s in f:
  s=s.rstrip()
  if s == '': break
  (term,tmp)=s.split()
  #print term, tmp
  qty=int(tmp.split(':')[1])
  dictList.append((-qty,term))
dictList.sort()

minFreq = 0
if len(dictList) >= maxWordQty:
  minFreq = -dictList[maxWordQty-1][0]

checkDict=dict()
for (qty,term) in dictList:
  if -qty >= minFreq:
    checkDict[term]=1
    #print "%s -> %d" % (term, -qty)

resDict=dict()
for line in sys.stdin:
  line=line.rstrip()
  if line == '' : continue
  tmp=line.split()
  wordOrig=tmp[0]
  word = wordOrig.lower()
  if not word in checkDict:
    continue
  lcFlag= word == wordOrig
  # If the word is in the lower-case we simply insert it (there shouldn't be another lower-case entry)
  # If there is a previously seen mixed-case entry, let's replace it
  if lcFlag:
    resDict[word]=line
  else:
  # However, there may be several mixed-case entries. We insert them only if there was no prior (in particular LOWER case entry!)
    if word in resDict:
      sys.stderr.write("Ignoring a word '%s' with upper case letetrs, because there is an entry already\n" % word)
      continue
    resDict[word]=line.lower()

for word in resDict:
  print resDict[word]
