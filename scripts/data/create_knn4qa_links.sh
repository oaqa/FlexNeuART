#!/bin/bash
function check {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    exit 1
  fi
}
DATA_ROOT=$1
if [ "$DATA_ROOT" = "" ] ; then
  echo "Specify the data directory root (1st arg)!"
  exit 1
fi
if [ ! -d "$DATA_ROOT" ] ; then
  echo "'$DATA_ROOT' is not a directory (1st arg)!"
  exit 1
fi
#lucene_index                                
ln -s "$DATA_ROOT/lucene_index"
check "ln -s "$DATA_ROOT/lucene_index""
#input                                
ln -s "$DATA_ROOT/input"
check "ln -s "$DATA_ROOT/input""
#memfwdindex                          
ln -s "$DATA_ROOT/memfwdindex"
check "ln -s "$DATA_ROOT/memfwdindex""
#output                               
ln -s "$DATA_ROOT/output/"
check "ln -s "$DATA_ROOT/output/""
#WordEmbeddings                       
ln -s "$DATA_ROOT/WordEmbeddings/"
check "ln -s "$DATA_ROOT/WordEmbeddings/""
#tran
mkdir -p tran
check "mkdir -p tran"
cd tran
check "cd tran"
ln -s "$DATA_ROOT/tran/ComprMinusManner" manner
check "ln -s "$DATA_ROOT/tran/ComprMinusManner" manner"
ln -s "$DATA_ROOT/tran/ComprMinusManner" ComprMinusManner
check "ln -s "$DATA_ROOT/tran/ComprMinusManner" ComprMinusManner"
ln -s "$DATA_ROOT/tran/compr" compr
check "ln -s "$DATA_ROOT/tran/compr" compr"
ln -s "$DATA_ROOT/tran/compr" compr2M
check "ln -s "$DATA_ROOT/tran/compr" compr2M"
