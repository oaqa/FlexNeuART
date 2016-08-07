#!/usr/bin/env python
import argparse
import gzip
import math
import numpy
import re
import sys
#
# This script is a modified version of https://github.com/mfaruqui/retrofitting
# Modifications:
#   1) Weighted retrofitting
#   2) Specifying the stop condition in terms of vector change, not the number of iteration!
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
  
''' Read the weighted relations as a dictionary '''
def read_lexicon(filename, wordVecs):
  lexicon = {}
  for line in open(filename, 'r'):
    allWords = line.lower().strip().split()
    mainWord = norm_word(allWords[0])
    words = []
    weights_ = []
    for k in range(1,len(allWords)):
      (word, weight) = allWords[k].split(':')
      word = norm_word(word)
      # We will ignore all words not present in the word embedding file
      if word in wordVecs:
        words.append(word)
        weights_.append(float(weight))
    weights = numpy.array(weights_)
    if (len(weights) >0): 
      weightsNorm = 1.0 / weights.sum()
      weights *= weightsNorm
    
    lexicon[mainWord] = [(words[k], weights[k]) for k in range(0, len(weights))]
  sys.stderr.write("Lexicon read from: "+filename+" \n")
  return lexicon

''' Retrofit word vectors to a lexicon '''
def retrofit(wordVecs, lexicon, alpha, eps, numIters):
  print "alpha=%f eps=%f" % (alpha, eps)
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = wvVocab.intersection(set(lexicon.keys()))
  dim=len(wordVecs[wordVecs.keys()[0]])
  divisor = 1.0/(1.0 + alpha)
  for it in range(numIters):
    diff=0
    # loop through every node also in ontology (else just use data estimate)
    for word in loopVocab:
      neighbors = lexicon[word]
      #no neighbours, pass - use data estimate
      neighbQty = len(neighbors)
      if neighbQty == 0:
        continue
      oldVec = deepcopy(newWordVecs[word])
      # Multiplying by alpha creates a new vector
      newVec = alpha * wordVecs[word]
      # loop over neighbors and add to new vector (currently with weight 1)
      for (ppWord, weight) in neighbors:
        newVec += weight * newWordVecs[ppWord]
      newWordVecs[word] = newVec * divisor
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
  parser.add_argument("-n", "--numiter", type=int, default=1000, help="Maximum number of iterations")
  parser.add_argument("-e", "--eps", type=float, default=1e-4, help="stopping threshold")
  parser.add_argument("-a", "--alpha", type=float, default=1.0, help="alpha-coefficient")
  args = parser.parse_args()

  wordVecs = read_word_vecs(args.input)
  lexicon = read_lexicon(args.lexicon, wordVecs)
  numIter = int(args.numiter)
  outFileName = args.output
  eps = args.eps
  alpha = args.alpha
  
  ''' Enrich the word vectors using ppdb and print the enriched vectors '''
  print_word_vecs(retrofit(wordVecs, lexicon, alpha, eps, numIter), outFileName) 
  print "Finished successfully!"
