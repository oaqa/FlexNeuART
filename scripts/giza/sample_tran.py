#!/usr/bin/env python
import random
import sys
import os

#
# This is a script that samples QA pairs
# (to later check how performance is affected by the amount of training data)
#

def Usage(err):
  if not err is None:
    print err
  print "Usage: <top level output directory> <collection name: e,g., compr, stackoverflow> <field name> <max power of 2 (we select roughly 2^{-k} QA pairs, where 2 <= k <= this parameter value)>"
  sys.exit(1)

def pow2(n):
  return 1<<n

def quest_file_name(d, fieldName):
  return d + '/question_' + fieldName

def answ_file_name(d, fieldName):
  return d + '/answer_' + fieldName

if len(sys.argv) != 5:
  Usage(None)

topLevelDir = sys.argv[1]
colName     = sys.argv[2]
fieldName   = sys.argv[3]
maxPower2    = int(sys.argv[4])

srcDir = topLevelDir + '/' + colName + '/tran'

if not os.path.isdir(srcDir):
  Usage("Cannot find source directory: '" + srcDir + "'")

nums = []
fSampleQuest = []
fSampleAnsw  = []

for i in range(1,maxPower2+1):
  n = pow2(i) 
  td = srcDir + str(n)
  os.mkdir(td)
  nums.append(n)
  fSampleQuest.append(open(quest_file_name(td, fieldName), 'w'))
  fSampleAnsw.append(open(answ_file_name(td, fieldName), 'w'))

fQuest = open(quest_file_name(srcDir, fieldName), 'r') 
fAnsw  = open(answ_file_name(srcDir, fieldName), 'r') 

ln=0

while True:
  ln = ln + 1
  lineQuest = fQuest.readline()
  if lineQuest == '' : 
    break
  lineAnsw = fAnsw.readline()
  if lineAnsw == '' :
    print "The answer file finished unexpectedly in line " + str(ln)
    sys.exit(1)
  lineQuest=lineQuest.strip()
  lineAnsw=lineAnsw.strip()
  if lineQuest == '' or lineAnsw == '' : continue
  for i in range(0, len(nums)):
    if random.randint(0, nums[i]-1) == 0:
      fSampleQuest[i].write(lineQuest + '\n')
      fSampleAnsw[i].write(lineAnsw + '\n')
  

for i in range(0, len(nums)):
  fSampleQuest[i].close()
  fSampleAnsw[i].close()




