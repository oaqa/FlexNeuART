#/bin/bash
. scripts/common.sh

col=$1
if [ "$col" = "" ] ; then
  echo "Specify collection name, e.g., squad (1st arg)"
  exit 1
fi

function one_set {
  dir=$1
  ndcg=`cat $1/@/rep/out_100.rep|grep -i NDCG|awk '{print $2}'`
  err=`cat $1/@/rep/out_100.rep|grep -i ERR|awk '{print $2}'`
  echo -e "$ndcg\t$err"
}

echo "BM25:"
one_set results.tune/feature_exper/$col/qrels_all_graded_same_score.txt/dev1//exper@bm25\=text+model1=text+minProbModel1\=text\:2.5e-3/

echo "BM25+Model1:"
for f in 2 4 8 16 32 64 128 256 ; do  
  one_set results.tran$f/feature_exper/$col/qrels_all_graded_same_score.txt/dev1/exper@bm25\=text+model1=text/
done
