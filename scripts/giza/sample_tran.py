#!/usr/bin/env python
import random
import sys
import os
import shutil
import random

#
# This is a script that samples QA pairs
# (to later check how performance is affected by the amount of training data)
#

def Usage(err):
  if not err is None:
    print(err)
  print("Usage: <top level output directory> <collection name: e,g., compr, stackoverflow>  <field name> " +
        "<src sub-dir name, e.g., tran> <dst sub-dir name, e.g., tran.sample> <# of entries to sample>")
  sys.exit(1)

def questFileName(dirName, fieldName):
  return os.path.join(dirName, 'question_' + fieldName)

def answFileName(dirName, fieldName):
  return os.path.join(dirName, 'answer_' + fieldName)

if len(sys.argv) != 7:
  Usage(None)

topLevelDir = sys.argv[1]
colName     = sys.argv[2]
fieldName   = sys.argv[3]
srcSubDir   = sys.argv[4]
dstSubDir   = sys.argv[5]
sampleQty   = int(sys.argv[6])

srcDir = os.path.join(topLevelDir, colName, srcSubDir)

if not os.path.isdir(srcDir):
  Usage("Cannot find source directory: '" + srcDir + "'")

dstDir = os.path.join(topLevelDir, colName, dstSubDir)

if os.path.exists(dstDir):
  shutil.rmtree(dstDir)

os.mkdir(dstDir)

fQuest = open(questFileName(dstDir, fieldName), 'w')
fAnsw = open(answFileName(dstDir, fieldName), 'w')

sampleQuest = []
sampleAnsw = []

readQty=0

while True:

  lineQuest = fQuest.readline()
  if lineQuest == '' : 
    break

  lineAnsw = fAnsw.readline()
  if lineAnsw == '' :
    print("The answer file finished unexpectedly in line " + str(readQty))
    sys.exit(1)

  lineQuest=lineQuest.strip()
  lineAnsw=lineAnsw.strip()
  if lineQuest == '' or lineAnsw == '' : continue

  readQty = readQty + 1

  if readQty <= sampleQty:  
    sampleQuest.append(lineQuest)
    sampleAnsw.append(lineAnsw)
  else:
    # randint is bounday-inclusive
    # a probability to include the last element is sampleQty / readQty
    x = random.randint(0, readQty - 1) == 0
    if x < sampleQty:
      sampleQuest[x] = lineQuest
      sampleAnsw[x] = lineAnsw

fSampleQuest = open(questFileName(dstDir, fieldName), 'w')
for lineQuest in sampleQuest:
  fSampleQuest.write(lineQuest + '\n')
fSampleQuest.close()

fSampleAnsw  = open(answFileName(dstDir, fieldName), 'w')
for lineAnsw in sampleAnsw:
  fSampleAnsw.write(lineAnsw + '\n')
fSampleAnsw.close()




