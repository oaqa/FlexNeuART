#!/usr/bin/env python3
import os, sys
import pickle
import matchzoo as mz
from matchzoo.data_pack import pack, DataPack
from matchzoo.preprocessors.basic_preprocessor import BasePreprocessor, BasicPreprocessor

from matchzoo.models.dssm import DSSM
from shutil import rmtree

sys.path.append('.')
from scripts.data.matchzoo_reader import *

colName = sys.argv[1]
modelFile = sys.argv[2]
epochQty = int(sys.argv[3])

dataTranFile = os.path.join('matchZooTrain', colName,  'data_transform.bin')
dataFileTrain = os.path.join('matchZooTrain', colName,  'tran_neg10.tsv')
dataFileTest = os.path.join('matchZooTrain', colName, 'dev1_allCand.tsv')

print(f'Collection: {colName} # of epochs: {epochQty} model file: {modelFile} data transform file: {dataTranFile}')


# Note dtype! don't let Pandas guess column data types!
dataTrainPacked = pack(readWhiteSpacedMatchZooData(dataFileTrain))
dataTestPacked = pack(readWhiteSpacedMatchZooData(dataFileTest))

#prep = mz.preprocessors.BasicPreprocessor()
prep = WhiteSpacePreprocessor()

import pdb, sys

#try:
if True:

  if os.path.exists(dataTranFile):
    print(f'Loading existing preprocessor from {dataTranFile}')
    with open(dataTranFile, 'rb') as f:
      prep = pickle.load(f) 
  else:
    print(f'Fitting a new preprocessor')

    # For some reason, we fit the preprocessor to packed data
    prep.fit(dataTrainPacked)

  print('Preprocessor context:')
  print(prep.context)

  with open(dataTranFile, 'wb') as of:
    pickle.dump(prep, of) 

  print('Data transformer is fitted and saved!')

  dataTrainProc = prep.transform(dataTrainPacked)
  dataTestProc = prep.transform(dataTestPacked)


  if os.path.exists(modelFile):
    print('Loading the model from: ' + modelFile)
    model = mz.load_model(modelFile)
    model.backend.summary()
  else:
    print('Creating a model from scratch')

    model = DSSM()
    model.params.update(prep.context)

    model.params['mlp_num_layers'] = 5
    model.params['mlp_num_units'] = 500
    model.params['mlp_num_fan_out'] = 128
    model.params['mlp_activation_func'] = 'relu'
    model.guess_and_fill_missing_params(verbose=0)
    print("Params completed",model.params.completed())
    model.build()

    model.compile()
    model.backend.summary()
  
  # This needs to use the processed data!
  xTrain, yTrain = dataTrainProc.unpack()
  model.fit(xTrain, yTrain, batch_size=128, epochs=epochQty)
  if os.path.exists(modelFile):
    rmtree(modelFile)
  model.save(modelFile)
  
#except:
   # tb is traceback
  #type, value, tb = sys.exc_info()
  #pdb.post_mortem(tb)

