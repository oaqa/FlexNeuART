#!/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

# Don't install Matchzoo packages by default
USE_MATCHZOO=0

plist=(\
lxml             "" \
torch            "1.4" \
torchtext        "" \
numpy            "" \
bs4              "" \
thrift           "0.13.0" \
spacy            "2.2.3" \
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

cd trec_eval
make
cd ..

if [ -d pytorch-pretrained-BERT-mod ] ; then
  rm -rf pytorch-pretrained-BERT-mod
fi

git clone https://github.com/searchivarius/pytorch-pretrained-BERT-mod
cd pytorch-pretrained-BERT-mod
python setup.py install
cd ..

qty=${#plist[*]}
for ((i=0;i<$qty;i+=2)) ; do
  pname=${plist[$i]}
  ver=${plist[$i+1]}
  echo "$pname -> $ver"
  if [ "$ver" = "" ] ; then
    pip install "$pname"
  else
    pip install "$pname==$ver"
  fi
done


python -m spacy download en_core_web_sm


echo "All is installed!"
