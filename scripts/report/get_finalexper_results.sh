#!/bin/bash
. scripts/common.sh
. scripts/report/report_common.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr (1st arg)"
  exit 1
fi

RESULTS_DIR="$2"
if [ "$RESULTS_DIR" = "" ] ; then
  echo "Specify a root for results directory (2d arg)"
  exit 1
fi
if [ ! -d "$RESULTS_DIR" ] ; then
  echo "Note a directory: $RESULTS_DIR (2d arg)"
  exit 1
fi

RUN_DESC_FILE="$3"
if [ "$RUN_DESC_FILE" = "" ] ; then
  echo "Specify a run description file (3d arg)"
  exit 1
fi
if [ ! -f "$RUN_DESC_FILE" ] ; then
  echo "Not a file (3d arg)"
  exit 1
fi

QREL_TYPE="$4"
QREL_FILE=`get_qrel_file "$QREL_TYPE" "4th"`
check ""

FILT_N="$5"
if [ "$FILT_N" = "" ] ; then
  FILT_N="*"
fi


EXPER_DIR="$RESULTS_DIR/final/"
TMP_FILE=`mktemp`
check "mktemp"


echo -e "method_label\ttop_k\tquery_time\tNDCG@20\tERR@20\tMAP\tMRR\trecall\tquery_num\tsuffix"

n=`wc -l "$RUN_DESC_FILE"|awk '{print $1}'`
n=$(($n+1))
for ((i=1;i<$n;++i))
  do
    line=`head -$i "$RUN_DESC_FILE"|tail -1`
    line=`echo $line|sed 's/#.*$//'|sed 's/^\s.*//'`
    if [ "$line" !=  "" ]
    then
      METHOD_LABEL=`echo $line|awk '{print $1}'`
      SUFFIX=`echo $line|awk '{print $2}'`
      TEST_SET=`echo $line|awk '{print $3}'`
      # Each experiment should run in its separate directory
      EXPER_DIR_UNIQUE="$EXPER_DIR/$collect/$QREL_FILE/$TEST_SET/$SUFFIX"
      if [ ! -d "$EXPER_DIR_UNIQUE" ] ; then
        echo "Directory doesn't exist: $EXPER_DIR_UNIQUE"
        exit 1
      fi
      pd=$PWD
      cd $EXPER_DIR_UNIQUE/rep >/dev/null
      check "cd $EXPER_DIR_UNIQUE/rep"

      # Let's read timings
      query_time="N/A"
      stat_file="stat_file"
      if [ -f "$stat_file" ] ; then 
        fn=`head -1 $stat_file|cut -f 1`
        if [ "$fn" != "QueryTime" ] ; then
          "Wrong format of the file (expecting that the first field is QueryTime, but got: '$fn'): $stat_file"
          exit 1
        fi
        query_time=`head -2 $stat_file|tail -1|cut -f 1`
        if [ "$query_time" = "" ] ; then
          "Cannot retrieve QueryTime from line 2 in the file: $stat_file"
          exit 1
        fi
      fi
      echo > $TMP_FILE
      for f in `ls -tr out_${FILT_N}.rep` ; do 
        top_k=`echo $f|sed 's/out_//'|sed 's/.rep//'`
        query_qty=`get_metric_value $f "# of queries"`
        ndcg20=`get_metric_value $f "NDCG@20"`
        err20=`get_metric_value $f "ERR@20"`
        p20=`get_metric_value $f "P@20"`
        map=`get_metric_value $f "MAP"`
        mrr=`get_metric_value $f "Reciprocal rank"`
        recall=`get_metric_value $f "Recall"`

        echo -e "$METHOD_LABEL\t$top_k\t$query_time\t$ndcg20\t$err20\t$map\t$mrr\t$recall\t$query_qty\t$SUFFIX" >> $TMP_FILE
      done
      sort -k 2,2 -n $TMP_FILE
      cd - >/dev/null
      check "cd $pd"
    fi
  done

rm $TMP_FILE
check "rm $TMP_FILE"
