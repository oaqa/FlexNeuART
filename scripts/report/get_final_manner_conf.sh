#!/bin/bash
#
# NOTE!!!! this script discards queries with zero binary recall (like Surdeanu et al 2011 did)
#
pct=$1
if [ "$pct" = "" ] ; then
  echo "Specify the percentile, e.g., 1,5 (1st arg)"
  exit 1
fi
REPORT_DIR="$2"
if [ "$REPORT_DIR" = "" ] ; then
  echo "Specify the output directory!"
  exit 1
fi
SKIP_RUN_GEN="$3"

ROUND_DIGIT=3

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

EXPER_REP_DIR="results/final/manner/test/lucene/exper@bm25=text+model1=text/rep/"
#out_10.trec_eval.bz2"

if [ "$SKIP_RUN_GEN" != "1" ] ; then
  echo "Generating data from runs"
  for N in 10 17 36 72
    do
      frep="$EXPER_REP_DIR/out_${N}.trec_eval.bz2"
      if [ ! -f $frep ] ; then
        echo "Missing file $frep"
      fi
      TREC_EVAL_FNAME="$REPORT_DIR/trec_eval_mod.$N"
      # Values for non-zero recall are obtained by retaining recors with non-zero reciprocal rank (RR) only
      bzcat $frep|awk '{print $1" "$2" "$3}'|grep -E '^recip_rank '|grep -Ev '(all|recip_rank .* 0\.00*$)'|awk '{vp1=0;if ($3>=1) vp1=1; print "recip_rank\t"$2"\t"$3; print "P_1\t"$2"\t"vp1;}' > $TREC_EVAL_FNAME
      REGISTRY_FILE="$REPORT_DIR/registry.$N"
      echo $TREC_EVAL_FNAME > $REGISTRY_FILE
      RUN_ROW="$REPORT_DIR/row.$N"
      for metric in P_1 recip_rank ; do
        scripts/report/conv_treceval.pl $metric $REGISTRY_FILE $RUN_ROW.$metric
        check "scripts/report/conv_treceval.pl $metric $REGISTRY_FILE $RUN_ROW.$metric"
      done

      TREC_EVAL_FNAME_ADD="$REPORT_DIR/trec_eval_mod_add.$N"
      echo $TREC_EVAL_FNAME_ADD > $REGISTRY_FILE
      bzcat $frep|awk '{print $1" "$2" "$3}'|grep -E '^num_rel_ret '|grep -v 'all' > $TREC_EVAL_FNAME_ADD
      for metric in num_rel_ret ; do
        scripts/report/conv_treceval.pl $metric $REGISTRY_FILE $RUN_ROW.$metric
        check "scripts/report/conv_treceval.pl $metric $REGISTRY_FILE $RUN_ROW.$metric"
      done
    done
else
  echo "Skipping run-data generation!"
fi

for metric in num_rel_ret P_1 recip_rank ; do
  echo "Metric $metric"
  echo "========================================"
  for N in 10 17 36 72
    do
      echo "N=$N"
      RUN_ROW="$REPORT_DIR/row.$N"
      scripts/report/conf-interv.R "$RUN_ROW.$metric" "$pct" "$ROUND_DIGIT"
      check "scripts/report/conf-inter.R ..."
    done
  echo "========================================"
done
