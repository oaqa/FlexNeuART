#!/usr/bin/env python3
import os, sys
import pickle
import matchzoo as mz
from matchzoo.data_pack import pack, DataPack

sys.path.append('.')
from scripts.data.matchzoo_reader import *

colName = sys.argv[1]
modelFile = sys.argv[2]

dataTranFile = os.path.join('matchZooTrain', colName,  'data_transform.bin')
dataFileTest = os.path.join('matchZooTrain', colName, 'dev1_allCand.tsv')

print(f'Collection: {colName}  model file: {modelFile} data transform file: {dataTranFile}')
print(f'Test file: {dataFileTest}')


# Note dtype! don't let Pandas guess column data types!
dataTestPacked = pack(readWhiteSpacedMatchZooData(dataFileTest))

with open(dataTranFile, 'rb') as f:
   prep = pickle.load(f) 

import pdb, sys

#try:
if True:
  dataTestProc = prep.transform(dataTestPacked)

  model=mz.load_model(modelFile)
  model.backend.summary()
  xTest, yTest = dataTestProc.unpack()
  model.params['task'].metrics = [mz.metrics.NormalizedDiscountedCumulativeGain(k=20)]
  print(model.evaluate(xTest, yTest, batch_size=128))
  
#except:
   # tb is traceback
  #type, value, tb = sys.exc_info()
  #pdb.post_mortem(tb)

