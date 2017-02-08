from scipy import sparse

#import sys,os
#sys.path.append(os.path.relpath("scripts/data"))
#from sparse_hlp import str2SparseVect,sparseVect2Str
def str2SparseVect(line):
  line=line.replace(':', ' ') 
  data=line.split() 
  vals=[]
  rind=[]
  cind=[]
  if len(data) % 2 != 0 : 
    raise Exception("Uneven # of elements in line: '" + line + "'")
  for i in range(0,len(data), 2):
    vals.append(float(data[i+1]))
    rind.append(0)
    cind.append(int(data[i]))
  return sparse.csr_matrix((vals,(tuple(rind),tuple(cind))))

def sparseVect2Str(oneRowMatr, numDig = 5):
  fStr="%."+str(numDig)+"f"
  nr=oneRowMatr.shape[0] 
  if nr != 1:
    raise Exception("The number of rows (" + nr + ") in the matrix is > 1");
  parts=[]
  ind=oneRowMatr.nonzero()
  cols=ind[1]
  for ci in cols:
    parts.append(str(ci)+':'+(fStr%(oneRowMatr[0,ci])))
  return ' '.join(parts)
