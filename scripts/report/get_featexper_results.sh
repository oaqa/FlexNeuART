#!/bin/bash
. scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "FEAT_EXPER_SUBDIR"

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr, squad (1st arg)"
  exit 1
fi

FEATURE_DESC_FILE="$2"
if [ "$FEATURE_DESC_FILE" = "" ] ; then
  echo "Specify a feature description file (2d arg)"
  exit 1
fi
if [ ! -f "$FEATURE_DESC_FILE" ] ; then
  echo "Not a file (2d arg)"
  exit 1
fi

FILT_N="$3"
if [ "$FILT_N" = "" ] ; then
  FILT_N="*"
fi

experDir="$COLLECT_ROOT/$collect/$FEAT_EXPER_SUBDIR"

echo -e "extractor_type\ttop_k\tquery_qty\tNDCG@20\tERR@20\tP@20\tMAP\tMRR\tRecall"

n=`wc -l "$FEATURE_DESC_FILE"|awk '{print $1}'`
n=$(($n+1))
for ((i=1;i<$n;++i))
  do
    line=`head -$i "$FEATURE_DESC_FILE"|tail -1`
    line=$(removeComment "$line")
    if [ "$line" !=  "" ]
    then
      # Each experiment should run in its separate directory
      extrType=`echo $line|awk '{print $1}'`
      testSet=`echo $line|awk '{print $2}'`
      experSubdir=`echo $line|awk '{print $3}'`

      experDirUnique=$(getExperDirUnique "$experDir" "$testSet" "$experSubdir")
      if [ ! -d "$experDirUnique" ] ; then
        echo "Directory doesn't exist: $experDirUnique"
        exit 1
      fi
      pd=$PWD
      cd $experDirUnique/rep
      check "cd $experDirUnique/rep"
      for f in `ls -tr out_${FILT_N}.rep` ; do 
        top_k=`echo $f|sed 's/out_//'|sed 's/.rep//'`
        query_qty=`get_metric_value "$f" "# of queries"`
        ndcg20=`get_metric_value "$f" "NDCG@20"`
        err20=`get_metric_value "$f" "ERR@20"`
        p20=`get_metric_value "$f" "P@20"`
        map=`get_metric_value "$f" "MAP"`
        mrr=`get_metric_value "$f" "Reciprocal rank"`
        recall=`get_metric_value "$f" "Recall"`
        echo -e "$extrType\t$top_k\t$query_qty\t$ndcg20\t$err20\t$p20\t$map\t$mrr\t$recall"
      done
      cd $pd
      check "cd $pd"
    fi
  done
