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
      line=`echo $line|sed 's/#.*$//'|sed 's/^\s.*//'`
    
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

        echo "Working with the directory: $EXPER_DIR_UNIQUE"

        RUN_ROW="$REPORT_DIR/$METHOD_LABEL.row"
        REGISTRY_FILE="$REPORT_DIR/$METHOD_LABEL.registry"

          frep="$EXPER_DIR_UNIQUE/rep/out_${FILT_N}.trec_eval.bz2"
          if [ ! -f $frep ] ; then
            echo "Missing file $frep"
            exit 1
          fi
          TREC_EVAL_FNAME="$REPORT_DIR/$METHOD_LABEL.trec_eval_mod"
          bzcat $frep|awk '{if ($2 != "all") {print $0;if ($1 == "num_rel") num_rel=$3;if ($1 == "num_rel_ret") {v=$3/num_rel;print "Recall\t"$2"\t"v;}}}' > $TREC_EVAL_FNAME
          echo $TREC_EVAL_FNAME > $REGISTRY_FILE
          for metric in "Recall" ; do
            scripts/report/conv_treceval.pl $metric $REGISTRY_FILE $RUN_ROW.$metric
            check "scripts/report/conv_treceval.pl $metric $REGISTRY_FILE $RUN_ROW.$metric"
          done

          frep="$EXPER_DIR_UNIQUE/rep/out_${FILT_N}.gdeval"
          if [ ! -f $frep ] ; then
            frep="$EXPER_DIR_UNIQUE/rep/out_${FILT_N}.gdeval.bz2"
            if [ ! -f $frep ] ; then
              echo "Missing file $frep"
              exit 1
            fi
            TREC_EVAL_FNAME="$REPORT_DIR/$METHOD_LABEL.gdeval_copy"
            bzcat $frep > $TREC_EVAL_FNAME
          else
            TREC_EVAL_FNAME=$frep
          fi
          echo $TREC_EVAL_FNAME > $REGISTRY_FILE
          for metric in "ndcg@20" "err@20" ; do
            scripts/report/conv_gdeval.pl $metric $REGISTRY_FILE $RUN_ROW.$metric
            check "scripts/report/conv_gdeval.pl $metric $REGISTRY_FILE $RUN_ROW.$metric"
          done
      fi
    done
  echo "Runs are generated!"
else
  echo "Skipping run-data generation!"
fi

echo "Computing p-values!"

for BASELINE_LABEL in "nmslib_bm25_text_brute_force" "lucene_bm25_model1" ; do
  echo "Baseline: $BASELINE_LABEL"

  for metric in "ndcg@20" "err@20" "Recall" ; do
    echo "Metric $metric"
    echo "========================================"
    for ((i=1;i<$n;++i))
      do
        line=`head -$i "$RUN_DESC_FILE"|tail -1`
        line=`echo $line|sed 's/#.*$//'|sed 's/^\s.*//'`

        METHOD_LABEL=`echo $line|awk '{print $1}'`

        if [ "$line" !=  "" -a "$METHOD_LABEL" != "$BASELINE_LABEL" ]
        then
          BASELINE_ROW="$REPORT_DIR/$BASELINE_LABEL.row.$metric"
          RUN_ROW="$REPORT_DIR/$METHOD_LABEL.row.$metric"

          SUFFIX=`echo $line|awk '{print $2}'`
          TEST_SET=`echo $line|awk '{print $3}'`


          EXPER_DIR_UNIQUE="$EXPER_DIR/$collect/$QREL_TYPE/$TEST_SET/$SUFFIX"
          if [ ! -d "$EXPER_DIR_UNIQUE" ] ; then
            echo "Directory doesn't exist: $EXPER_DIR_UNIQUE"
            exit 1
          fi

          # Let's read timings
          query_time="N/A"
          stat_file="$EXPER_DIR_UNIQUE/rep/stat_file"
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
      

          scripts/report/t-test.R "$RUN_ROW" "$BASELINE_ROW" "$RUN_QTY" 0.01 "$metric/$BASELINE_LABEL"
          check "scripts/report/t-test.R \"$RUN_ROW\" \"$BASELINE_ROW\" \"$RUN_QTY\" 0.01" "$metric/$BASELINE_LABEL"
          echo "Query time: $query_time"
          echo "----------------------------"
        fi
    done
    echo "========================================"

  done
done

echo "P-values are computed!"
