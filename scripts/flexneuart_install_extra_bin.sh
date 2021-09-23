#!/bin/bash -e
set -o pipefail
echo "Installing extra binaries into the current directory!"
withGiza="$1"
[ ! -z "$withGiza" ]  || { echo "Specify the MGIZA installation flag (1st arg)" ; exit 1 ; }

cd "$dstDir"
curr_dir=$PWD

python -m spacy download en_core_web_sm

TREC_EVAL_VER="9.0.7"
trec_file_pref=v${TREC_EVAL_VER}
rm -f "${trec_file_pref}.tar.gz"
wget https://github.com/usnistgov/trec_eval/archive/refs/tags/${trec_file_pref}.tar.gz
tar zxvf ${trec_file_pref}.tar.gz
rm -f trec_eval
ln -s trec_eval-${TREC_EVAL_VER} trec_eval
rm "${trec_file_pref}.tar.gz"

cd $curr_dir/trec_eval
make

if [ "$withGiza" = "1" ] ; then
  pip install cmake

  cd $curr_dir
  rm -rf mgiza
  git clone https://github.com/moses-smt/mgiza.git
  cd mgiza/mgizapp
  cmake .
  make -j 4
  make install
fi

echo "Extra binaries are installed!"
