#!/usr/bin/env python
import sys

if len(sys.argv) != 3:
  print "Usage: <desired weight> <rank>"
  sys.exit(1)

desWght=float(sys.argv[1])
rank=int(sys.argv[2])

def weight(p, rank): 
  from math import log, pow
  q=1-p
  return 1 - pow(p, rank - 1) + q/p*rank*(-log(q)-sum([pow(p, i)/i for i in range(1,rank)]))

pMax=1e-8
pMin=1-pMax # The higher is p the smaller is the weight

wMin=weight(pMin, rank)
wMax=weight(pMax, rank)

if desWght <= wMin:
  print "The desired weight is too small!"
  sys.exit(1)

if desWght >= wMax:
  print "The desired weight is too large!"
  sys.exit(1)

epsTol=1e-4
while abs(wMin-wMax) > epsTol :
  p = 0.5*(pMin+pMax)
  w = weight(p, rank)
  if w < desWght :
    wMin = w
    pMin = p 
  else:
    wMax = w
    pMax = p
  
print "Probability: %f weight %f" % (p, w)
