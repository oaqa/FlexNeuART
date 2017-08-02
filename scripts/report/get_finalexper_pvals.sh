#!/bin/bash
. scripts/common.sh

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr (1st arg)"
  exit 1
fi

QREL_TYPE=`get_qrel_file $2 "2d"`

echo "qrel types: $QREL_TYPE"

RUN_DESC_FILE="$3"
if [ "$RUN_DESC_FILE" = "" ] ; then
  echo "Specify a run description file (3d arg)"
  exit 1
fi
if [ ! -f "$RUN_DESC_FILE" ] ; then
  echo "Not a file (3d arg)"
  exit 1
fi

FILT_N="$4"
if [ "$FILT_N" = "" ] ; then
  echo "Specify the depth of the candidate pool (4th arg)"
  exit 1
fi
REPORT_DIR="$5"
if [ "$REPORT_DIR" = "" ] ; then
  echo "Specify the output directory! (5th arg)"
  exit 1
fi
SKIP_RUN_GEN="$6"

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

mkdir -p $REPORT_DIR
check "to create an output dir '$REPORT_DIR'"

EXPER_DIR="results/final/"

RUN_QTY=`wc -l "$RUN_DESC_FILE"|awk '{print $1}'`
n=$(($RUN_QTY+1))
if [ "$SKIP_RUN_GEN" != "1" ] ; then
  echo "Generating data from $RUN_QTY runs"
  for ((i=1;i<$n;++i))
    do
      line=`head -$i "$RUN_DESC_FILE"|tail -1`
      if [ "$line" !=  "" ]
      then
        METHOD_LABEL=`echo $line|awk '{print $1}'`
        SUFFIX=`echo $line|awk '{print $2}'`
        TEST_SET=`echo $line|awk '{print $3}'`
        # Each experiment should run in its separate directory
        EXPER_DIR_UNIQUE="$EXPER_DIR/$collect/$QREL_TYPE/$TEST_SET/$SUFFIX"
        if [ ! -d "$EXPER_DIR_UNIQUE" ] ; then
          echo "Directory doesn't exist: $EXPER_DIR_UNIQUE"
          exit 1
        fi
        frep="$EXPER_DIR_UNIQUE/rep/out_${FILT_N}.trec_eval.bz2"
        if [ ! -f $frep ] ; then
          echo "Missing file $frep"
        fi
        TREC_EVAL_FNAME="$REPORT_DIR/$METHOD_LABEL.trec_eval_mod"
        # This line simply computes P_1, which is not present in the original data
        bzcat $frep|awk '{if ($2 != "all") {if ($1 != "recip_rank") print $0;else {v=0;if ($3>=1) v=1; print "P_1\t"$2"\t"v;}}}' > $TREC_EVAL_FNAME
        REGISTRY_FILE="$REPORT_DIR/$METHOD_LABEL.registry"
        echo $TREC_EVAL_FNAME > $REGISTRY_FILE
        RUN_ROW="$REPORT_DIR/$METHOD_LABEL.row"
        for metric in NDCG@20 ERR@20 Recall ; do
          scripts/report/conv_treceval.pl $metric $REGISTRY_FILE $RUN_ROW.$metric
          check "scripts/report/conv_treceval.pl $metric $REGISTRY_FILE $RUN_ROW.$metric"
        done
      fi
    done
else
  echo "Skipping run-data generation!"
fi

BM25_LABEL="nmslib_bm25_text_brute_force"
for metric in NDCG@20 ERR@20 Recall ; do
  echo "Metric $metric"
  echo "========================================"
  for ((i=1;i<$n;++i))
    do
      line=`head -$i "$RUN_DESC_FILE"|tail -1`
      if [ "$line" !=  "" ]
      then
        METHOD_LABEL=`echo $line|awk '{print $1}'`
        BM25_ROW="$REPORT_DIR/$BM25_LABEL.row.$metric"
        RUN_ROW="$REPORT_DIR/$METHOD_LABEL.row.$metric"

        scripts/report/t-test.R "$RUN_ROW" "$BM25_ROW" "$RUN_QTY"
        check "scripts/report/t-test.R ..."
      fi
  done
  echo "========================================"
done
