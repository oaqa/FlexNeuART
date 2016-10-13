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
INSTALL_DIR=$2
if [ "$INSTALL_DIR" = "" ] ; then
  echo "Specify the directory where our software is installed (including clones of repositories knn4qa and nmslib) (2d arg)"
  exit 1
fi
if [ ! -d "$INSTALL_DIR" ] ; then
  echo "Not a directory: $INSTALL_DIR"
  exit 1
fi
KNN4QA_DIR="$INSTALL_DIR/knn4qa"
if [ ! -d "$INSTALL_DIR" ] ; then
  echo "Not a directory: $INSTALL_DIR"
  exit 1
fi

collect=$3
if [ "$collect" = "" ] ; then
  echo "Specify collection name (3d arg): manner, compr, stackoverflow, ..."
  exit 1
fi
tran_src_name="$collect"
if [ "$collect" = "manner" ] ; then
  tran_src_name="ComprMinusManner"
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
# Headers and/or models may not be always present
if [ -d "$KNN4QA_DIR/scripts/nmslib/meta/compr/headers/"  ] ; then
  ln -s "$KNN4QA_DIR/scripts/nmslib/meta/compr/headers/" 
  check "ln -s $KNN4QA_DIR/scripts/nmslib/meta/compr/headers/" 
fi
if [ -d "$KNN4QA_DIR/scripts/nmslib/meta/compr/models/"  ] ; then
  ln -s "$KNN4QA_DIR/scripts/nmslib/meta/compr/models/" 
  check "ln -s $KNN4QA_DIR/scripts/nmslib/meta/compr/models/" 
fi
