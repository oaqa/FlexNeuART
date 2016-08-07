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
# Should be run from nmslib dir
DATA_ROOT=$1
if [ "$DATA_ROOT" = "" ] ; then
  echo "Specify the data directory root (1st arg)!"
  exit 1
fi
if [ ! -d "$DATA_ROOT" ] ; then
  echo "'$DATA_ROOT' is not a directory (1st arg)!"
  exit 1
fi
collect=$2
if [ "$collect" = "" ] ; then
  echo "Specify collection name (2d arg): manner, compr2M, or compr"
  exit 1
fi
tran_src_name="$collect"
if [ "$collect" = "manner" ] ; then
  tran_src_name="ComprMinusManner"
elif [ "$collect" = "compr2M" -o "$collect" = "compr" ] ; then
  tran_src_name="compr"
else
  echo "Unknown collection name: $collect (expect manner, compr2M, or compr)"
  exit 1
fi
if [ ! -d "$collect" ] ; then
  echo "Not a directory: '$collect'"
  echo "Please, double check that this script is executed in the directory with collection sub-directories"
  exit 1
fi
cd "$collect"
if [ "$?" != "0" ] ; then
  echo "Failed to change directory to '$collect" 
  exit 1
fi
ln -s "$DATA_ROOT/tran/$tran_src_name" tran
check "ln -s "$DATA_ROOT/tran/$tran_src_name" tran"
ln -s "$DATA_ROOT/memfwdindex/$collect" memfwdindex
check "ln -s "$DATA_ROOT/memfwdindex/$collect" memfwdindex"
ln -s "$DATA_ROOT/WordEmbeddings/$collect" WordEmbeddings
check "ln -s "$DATA_ROOT/WordEmbeddings/$collect" WordEmbeddings"
