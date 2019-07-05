#!/usr/bin/env python
import sys
if len(sys.argv) != 7:
  print("Usage: <questions> <answers> <maximum fertility> <output questions> <output answers> <symmetrize? 1 or 0>")
questFile=open(sys.argv[1], 'r')
answFile=open(sys.argv[2], 'r')
maxFert = int(sys.argv[3])
outFileQuest=open(sys.argv[4], "w")
outFileAnsw=open(sys.argv[5], "w")
symmetr=int(sys.argv[6]) != 0
print("Symmetrizing %d, maxFet %d" % (symmetr, maxFert))
for quest in questFile:
  answ = answFile.readline()
  len1=len(answ.split())
  len2=len(quest.split())
  if len2 <= len1*maxFert and len1 <= len2*maxFert:
    outFileQuest.write(quest)
    outFileAnsw.write(answ)
    if symmetr:
      outFileQuest.write(answ)
      outFileAnsw.write(quest)
    
