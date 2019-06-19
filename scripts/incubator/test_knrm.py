#!/usr/bin/env python3
import os, sys
import pickle
import matchzoo as mz
from matchzoo.data_pack import pack, DataPack

from matchzoo.models.knrm import KNRM

sys.path.append('.')
from scripts.data.matchzoo_reader import *

colName = sys.argv[1]
modelFilePrefix = sys.argv[2]
modelFile = modelFilePrefix + '.bin'
dataPrepFile = modelFilePrefix + '.dtran'

print(f'Collection: {colName} # model prefix: {modelFilePrefix}')

dataFileTest= os.path.join('matchZooTrain', colName, 'dev1_allCand.tsv')

# Note dtype! don't let Pandas guess column data types!
dataTestPacked = pack(readWhiteSpacedMatchZooData(dataFileTest))

with open(dataPrepFile, 'rb') as f:
   prep = pickle.load(f) 

import pdb, sys

#try:
if True:
  dataTestProc = prep.transform(dataTestPacked)

  model=mz.load_model(modelFile)
  xTest, yTest = dataTestProc.unpack()
  model.params['task'].metrics = [mz.metrics.NormalizedDiscountedCumulativeGain(k=20)]
  print(model.evaluate(xTest, yTest, batch_size=128))
  
#except:
   # tb is traceback
  #type, value, tb = sys.exc_info()
  #pdb.post_mortem(tb)

