#!/usr/bin/env python
import argparse
import gzip
import math
import numpy
import re
import sys
#
# This script is a slightly modified version of https://github.com/mfaruqui/retrofitting
# Modification: Specifying the stop condition in terms of vector change, not the number of iteration!
#
# Created by Manaal Faruqui, mfaruqui@cs.cmu.edu
# If you use it, please, cite the corresponding paper:
# @InProceedings{faruqui:2014:NIPS-DLRLW,
#  author    = {Faruqui, Manaal and Dodge, Jesse and Jauhar, Sujay K.  and  Dyer, Chris and Hovy, Eduard and Smith, Noah A.},
#  title     = {Retrofitting Word Vectors to Semantic Lexicons},
#  booktitle = {Proceedings of NAACL},
#  year      = {2015},
#} 
#

from copy import deepcopy

isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()

'''Euclidean distance'''
def eucl(x,y):
  return math.sqrt(((x-y)**2).sum())

''' Read all the word vectors and normalize them '''
def read_word_vecs(filename):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')
  
  for line in fileObject:
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
    
  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors

''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
  sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
  outFile = open(outFileName, 'w')  
  for word, values in wordVectors.iteritems():
    outFile.write(word+' ')
    for val in wordVectors[word]:
      outFile.write('%.4f' %(val)+' ')
    outFile.write('\n')      
  outFile.close()
  
''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename, wordVecs):
  lexicon = {}
  for line in open(filename, 'r'):
    words = line.lower().strip().split()
    lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
  return lexicon

''' Retrofit word vectors to a lexicon '''
def retrofit(wordVecs, lexicon, eps, numIters):
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = wvVocab.intersection(set(lexicon.keys()))
  dim=len(wordVecs[wordVecs.keys()[0]])
  for it in range(numIters):
    diff=0
    # loop through every node also in ontology (else just use data estimate)
    for word in loopVocab:
      wordNeighbours = set(lexicon[word]).intersection(wvVocab)
      numNeighbours = len(wordNeighbours)
      #no neighbours, pass - use data estimate
      if numNeighbours == 0:
        continue
      # the weight of the data estimate if the number of neighbours
      oldVec = deepcopy(newWordVecs[word])
      newVec = numNeighbours * wordVecs[word]
      # loop over neighbours and add to new vector (currently with weight 1)
      for ppWord in wordNeighbours:
        newVec += newWordVecs[ppWord]
      newWordVecs[word] = newVec/(2*numNeighbours)
      diff += eucl(newWordVecs[word], oldVec)
    print "Iteration: %d dim=%d change: %f" % (it, dim, diff)
    if diff < eps:
      print "Stopping threshold reached"
      break
  return newWordVecs
  
if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
  parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
  parser.add_argument("-o", "--output", type=str, help="Output word vecs")
  parser.add_argument("-n", "--numiter", type=int, default=1000, help="Num iterations")
  parser.add_argument("-e", "--eps", type=float, default=1e-4, help="stopping threshold")
  args = parser.parse_args()

  wordVecs = read_word_vecs(args.input)
  lexicon = read_lexicon(args.lexicon, wordVecs)
  numIter = int(args.numiter)
  outFileName = args.output
  eps = args.eps
  
  ''' Enrich the word vectors using ppdb and print the enriched vectors '''
  print_word_vecs(retrofit(wordVecs, lexicon, eps, numIter), outFileName) 
  print "Finished successfully!"
