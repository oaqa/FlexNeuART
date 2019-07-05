#!/usr/bin/env python3
import os, sys
import pickle
import matchzoo as mz
from matchzoo.data_pack import pack, DataPack
from matchzoo.preprocessors.basic_preprocessor import BasePreprocessor, BasicPreprocessor

from matchzoo.models.knrm import KNRM

sys.path.append('.')
from scripts.data.matchzoo_reader import *

colName = sys.argv[1]
modelFile = sys.argv[2]
epochQty = int(sys.argv[3])

dataTranFile = os.path.join('matchZooTrain', colName,  'data_transform.bin')
dataFileTrain = os.path.join('matchZooTrain', colName,  'tran_neg10.tsv')
dataFileTest = os.path.join('matchZooTrain', colName, 'dev1_allCand.tsv')

print(f'Collection: {colName} # of epochs: {epochQty} model file: {modelFile} data transform file: {dataTranFile}')

if os.path.exists(modelFile):
  # Stupid hack for now, b/c save will fail if the model exists
  print('Model already exists, exiting!')
  sys.exit(1)

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

  model=KNRM()
  model.params.update(prep.context)

  model.params['embedding_input_dim'] =  10000
  model.params['embedding_output_dim'] =  10
  model.params['embedding_trainable'] = True
  model.params['kernel_num'] = 11
  model.params['sigma'] = 0.1
  model.params['exact_sigma'] = 0.001

  model.guess_and_fill_missing_params(verbose=1)
  print("Params completed",model.params.completed())
  model.build()
  model.compile()
  model.backend.summary()
  
  # This needs to use the processed data!
  xTrain, yTrain = dataTrainProc.unpack()
  model.fit(xTrain, yTrain, batch_size=128, epochs=epochQty)
  model.save(modelFile)
  xTest, yTest = dataTestProc.unpack()
  print(model.evaluate(xTest, yTest, batch_size=128))
  
#except:
   # tb is traceback
  #type, value, tb = sys.exc_info()
  #pdb.post_mortem(tb)

