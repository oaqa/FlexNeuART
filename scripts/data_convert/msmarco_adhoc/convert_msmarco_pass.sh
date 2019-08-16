#!/bin/bash -e
# The main script to convert MSMARCO passage collection 
# It should be invoked after a bit of data massaging
# After downloading data from https://microsoft.github.io/TREC-2019-Deep-Learning/
# 1. documents needs to recompressed to collection.tsv.gz
# 2. queries need to be decompressed
source scripts/common_proc.sh
src=$1
if [ "$src" = "" ] ; then
  echo "Specify the source directory (1st arg)"
  exit 1
fi
dst=$2
if [ "$dst" = "" ] ; then
  echo "Specify the target directory (2d arg)"
  exit 1
fi

mkdir -p $dst/input_data/test2019
python -u scripts/data_convert/msmarco_adhoc/convert_queries.py --input  $src/msmarco-test2019-queries.tsv --output $dst/input_data/test2019/QuestionFields.jsonl

for part in pass train dev eval ; do
  mkdir -p $dst/input_data/$part
done

for part in train dev ; do
  cp $src/qrels.$part.tsv $dst/input_data/$part/qrels.txt
done
for part in train dev eval ; do
  python -u scripts/data_convert/msmarco_adhoc/convert_queries.py --input  $src/queries.$part.tsv --output $dst/input_data/$part/QuestionFields.jsonl
done

python -u scripts/data_convert/msmarco_adhoc/convert_pass.py --input $src/collection.tsv.gz  --output $dst/input_data/pass/AnswerFields.jsonl.gz 

