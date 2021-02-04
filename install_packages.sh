#!/bin/bash -e
set -o pipefail

curr_dir=$PWD

source scripts/common_proc.sh
source scripts/config.sh

# Don't install Matchzoo packages by default
USE_MATCHZOO=0

plist=(\
lxml             "" \
pytools          "" \
torch            "1.4" \
torchtext        "0.6.0" \
numpy            "" \
bs4              "" \
thrift           "0.13.0" \
spacy            "2.2.3" \
pyjnius          ""
)

if [ "$USE_MATCHZOO" = "1" ] ; then
  echo "Adding matchzoo packages"
  plist+=(\
          matchzoo         "" \
          keras            "2.3.0" \
          tensorflow       "")
fi

echo "Python packages to install:"
echo ${plist[*]}

if [ -d pytorch-pretrained-BERT-mod ] ; then
  rm -rf pytorch-pretrained-BERT-mod
fi
cd $curr_dir

qty=${#plist[*]}
for ((i=0;i<$qty;i+=2)) ; do
  pname=${plist[$i]}
  ver=${plist[$i+1]}
  echo "Installing package $pname -> $ver"
  if [ "$ver" = "" ] ; then
    pip install "$pname"
  else
    pip install "$pname==$ver"
  fi
done


python -m spacy download en_core_web_sm

# This should be installed after numpy or else it will try to isntall an incompatible version!
cd $curr_dir
git clone https://github.com/searchivarius/pytorch-pretrained-BERT-mod
cd pytorch-pretrained-BERT-mod
python setup.py install

cd $curr_dir
rm -rf mgiza
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp
cmake .
make -j 4
make install

cd $curr_dir/trec_eval 
make  

cd $curr_dir
cd lemur-code-r2792-RankLib-trunk >/dev/null
mvn clean package 
cp target/RankLib-2.14.jar ../lib/umass/RankLib/2.14.fixed/RankLib-2.14.fixed.jar

echo "All is installed!"
