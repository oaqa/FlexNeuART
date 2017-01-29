#!/bin/bash
. scripts/common.sh

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
IS_GALAGO_EXPER="0"
echo "$FEATURE_DESC_FILE" | grep galago >/dev/null
if [ "$?" = "0" ] ; then
  IS_GALAGO_EXPER="0"
fi

QREL_TYPE="$3"
QREL_FILE=`get_qrel_file "$QREL_TYPE" "3rd"`
check ""

FILT_N="$4"

if [ "$FILT_N" = "" ] ; then
  FILT_N="*"
fi

if [ "$IS_GALAGO_EXPER" = "0" ] ; then 
  EXPER_DIR="results/feature_exper/"
else
  EXPER_DIR="results/galago_exper/"
fi

echo -e "extractor_type\tembed_list\ttop_k\tquery_qty\tNDCG@20\tERR@20\tP@20\tMAP\tMRR\tRecall"

function get_param_value {
  f=$1
  pname=$2
  grep "$pname" $f|sed 's/ //g'|cut -d : -f 2
}

n=`wc -l "$FEATURE_DESC_FILE"|awk '{print $1}'`
n=$(($n+1))
for ((i=1;i<$n;++i))
  do
    line=`head -$i "$FEATURE_DESC_FILE"|tail -1`
    if [ "$line" !=  "" ]
    then
      # Each experiment should run in its separate directory
      if [ "$IS_GALAGO_EXPER" = "0" ] ; then 
        EXTR_TYPE=`echo $line|awk '{print $1}'`
        EMBED_LIST=`echo $line|awk '{print $2}'`
        TEST_SET=`echo $line|awk '{print $3}'`
        suffix="$EXTR_TYPE/$EMBED_LIST"
      else
        GALAGO_OP=`echo $line|awk '{print $1}'`
        GALAGO_PARAMS=`echo $line|awk '{print $2}'`
        TEST_SET=`echo $line|awk '{print $3}'`
        suffix="$GALAGO_OP/$GALAGO_PARAMS"
      fi
      EXPER_DIR_UNIQUE="$EXPER_DIR/$collect/$QREL_FILE/$TEST_SET/$suffix"
      if [ ! -d "$EXPER_DIR_UNIQUE" ] ; then
        echo "Directory doesn't exist: $EXPER_DIR_UNIQUE"
        exit 1
      fi
      pd=$PWD
      cd $EXPER_DIR_UNIQUE/rep
      check "cd $EXPER_DIR_UNIQUE/rep"
      for f in `ls -tr out_${FILT_N}.rep` ; do 
        top_k=`echo $f|sed 's/out_//'|sed 's/.rep//'`
        query_qty=`get_param_value $f "# of queries"`
        ndcg20=`get_param_value $f "NDCG@20"`
        err20=`get_param_value $f "ERR@20"`
        p20=`get_param_value $f "P@20"`
        map=`get_param_value $f "MAP"`
        mrr=`get_param_value $f "Reciprocal rank"`
        recall=`get_param_value $f "Recall"`
        echo -e "$EXTR_TYPE\t$EMBED_LIST\t$top_k\t$query_qty\t$ndcg20\t$err20\t$p20\t$map\t$mrr\t$recall"
      done
      cd $pd
      check "cd $pd"
    fi
  done
