#!/bin/bash -e
set -o pipefail

withGiza="$1"
[ "$withGiza" != "" ] || { echo "Empty GIZA flag (1st argument)" ;  exit 1 ; }

curr_dir=$PWD

# 1. Unpack scripts
REPO=$(python -c "from flexneuart import get_jars_location ; print(get_jars_location())") || { echo "import error, did you install flexneuart library?" ; exit 1 ; }
tar zxvf "$REPO/../scripts.tar.gz"

python -m spacy download en_core_web_sm 

cd $curr_dir

if [ "$withGiza" = "1" ] ; then
  cd $curr_dir
  rm -rf mgiza
  git clone https://github.com/moses-smt/mgiza.git 
  cd mgiza/mgizapp 
  cmake . 
  make -j 4 
  make install 
fi

TREC_EVAL_VER=9.0.7
rm -f v$TREC_EVAL_VER.tar.gz
wget https://github.com/usnistgov/trec_eval/archive/refs/tags/v$TREC_EVAL_VER.tar.gz 
rm -rf trec_eval-$TREC_EVAL_VER 
tar xvfj v$TREC_EVAL_VER.tar.gz
ln -s trec_eval-$TREC_EVAL_VER trec_eval

cd $curr_dir/trec_eval  
make 

echo "All is done!"
