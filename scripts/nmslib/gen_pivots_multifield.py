#!/usr/bin/env python
import sys
import random
import argparse

class BetterParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = BetterParser()
parser.add_argument("-d", "--index_dir",       type=str, required=True, help="A root directory for forward indices") 
parser.add_argument("-o", "--output_file",     type=str, required=True, help="An output file") 
parser.add_argument("-q", "--pivot_qty",       type=int, required=True, help="A number of pivots to generate")
parser.add_argument("-f", "--fields",          type=str, required=True, help="Input forward indices (comma-separated list), e.g., text,text_unlemm,bigram")
parser.add_argument("-m", "--max_term_qty",    type=str, required=True, help="Maximum number of most frequent terms to use (comma-separated list), e.g., 50000,50000,100000")
parser.add_argument("-t", "--pivot_term_qty",    type=str, required=True, help="Number of terms PER PIVOT (comma-separated list), e.g., 300,300,5000")

args = parser.parse_args()

inputDir=args.index_dir
outFile=args.output_file
pivotQty=args.pivot_qty
fieldList = args.fields.split(',')
maxTermQtyList = [int(x) for x in args.max_term_qty.split(',')]
pivotTermQtyList = [int(x) for x in args.pivot_term_qty.split(',')]

if len(fieldList) != len(maxTermQtyList) or len(fieldList) != len(pivotTermQtyList):
  print "The number of elements should be equal in the lists of input indices, maximum number of terms to use, maximum number of terms per PIVOT"
  sys.exit(1)

inputFiles= ["%s/%s" % (inputDir, x) for x in fieldList] 

print "We will generate %d pivots and save them to the file %s" % (pivotQty, outFile)

fieldDicts=[]
for fieldId in range(0, len(inputFiles)):
  oneFieldDict  =[]
  fname         = inputFiles[fieldId]
  maxTermQty    = maxTermQtyList[fieldId]
  pivotTermQty  = pivotTermQtyList[fieldId]
  
  print "Field %s: each pivot will contain %d terms selected among %d most frequent ones" % (fieldList[fieldId], pivotTermQty, maxTermQty)
  
  if pivotTermQty and maxTermQty > 0:
    f=open(fname)

    f.readline()
    s=f.readline().rstrip()
    if s != '': raise Exception("We expect the second line to be empty!")
    pairList = []
    for s in f:
      s=s.rstrip()
      if s == '': break
      (term,tmp)=s.split()
      (termId,freq)=tmp.split(':')
      freq=int(freq)
      pairList.append((-freq, termId))

    pairList.sort()
    minFreq=0
    if len(pairList) > maxTermQty:
      minFreq = -pairList[maxTermQty-1][0]
    print "A minimum frequency for field %s is %d" % (fieldList[fieldId],minFreq)
    for (freq,termId) in pairList:
      if (-freq) >= minFreq:
        oneFieldDict.append(int(termId))

  print "Selected %d dict. entries from %s" % (len(oneFieldDict), fname)
  fieldDicts.append(oneFieldDict)

fout=open(outFile, 'w')
fout.write('isQueryFile=1\n\n')
for i in range(0, pivotQty):
  # This is the line for IDs (IDs aren't present in queries)
  for fieldId in range(0, len(fieldDicts)):
    pivotTermQty = pivotTermQtyList[fieldId]
    oneFieldDict = fieldDicts[fieldId]
    if len(oneFieldDict) == 0: # just skip this
      fout.write('\n\n');
    else:
      pivArr=[]
      seen={}
      for k in range(0, pivotTermQty):
        indx = random.randint(0, len(oneFieldDict)-1) 
        while indx in seen:
          indx = random.randint(0, len(oneFieldDict)-1) 
        seen[indx]=1
        pivArr.append(oneFieldDict[indx])
      pivArr.sort()
      pivArrMod =[str(pivId) + ':1' for pivId in pivArr]
      # One empty line is for the sequence of word IDs, which we don't provide here
      fout.write(' '.join(pivArrMod) + '\n\n');
  fout.write('\n');

  if (i+1) % 100 == 0: print "# of pivots %d" % (i+1)
  
fout.close()
  
  
