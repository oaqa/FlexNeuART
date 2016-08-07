#!/bin/bash
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

function check_pipe {
  f="${PIPESTATUS[*]}"
  name=$1
  if [ "$f" != "0 0" ] ; then
    echo "******************************************"
    echo "* Failed (pipe): $name, exit statuses: $f "
    echo "******************************************"
    exit 1
  fi
}

POS_ARGS=()

NUM_CPU_CORES=`scripts/exper/get_cpu_cores.py`
check "getting the number of CPU cores, do you have /proc/cpu/info?"
THREAD_QTY=$NUM_CPU_CORES

NMSLIB_ADDR=""
NMSLIB_SPACE="qa1"

nmslib_fields_param=""
extr_type_interm_param=""
model_interm_param=""
cand_qty_param=""
giza_expand_qty_param=""
giza_wght_expand_param=""
delete_trec_runs="1"

while [ $# -ne 0 ] ; do
  echo $1|grep "^-" >/dev/null 
  if [ $? = 0 ] ; then
    OPT_NAME="$1"
    if [ "$OPT_NAME" = "-dont_delete_trec_runs" ] ; then
      # option without an argument
      shift 1
    elif [ "$OPT_NAME" = "-giza_wght_expand" ] ; then
      # option without an argument
      shift 1
    else
      OPT_VALUE="$2"
      OPT="$1 $2"
      if [ "$OPT_VALUE" = "" ] ; then  
        echo "Option $OPT_NAME requires an argument." >&2
        exit 1
      fi
      shift 2
    fi
    case $OPT_NAME in
      -thread_qty)
        THREAD_QTY=$OPT_VALUE 
        ;;
      -max_num_query)
        max_num_query_param=$OPT
        ;;
      -nmslib_addr)
        NMSLIB_ADDR=$OPT_VALUE
        ;;
      -nmslib_fields)
        nmslib_fields_param=$OPT
        ;;
      -model_interm)
        model_interm_param=$OPT
        ;;
      -giza_expand_qty)
        giza_expand_qty_param=$OPT
        ;;
      -giza_wght_expand)
        giza_wght_expand_param=$OPT_NAME
       ;;
      -extr_type_interm)
        extr_type_interm_param=$OPT
       ;; 
      -cand_qty)
        cand_qty_param=$OPT
       ;; 
      -dont_delete_trec_runs)
        delete_trec_runs="0"
       ;; 
      *)
        echo "Invalid option: $OPT_NAME" >&2
        exit 1
        ;;
    esac
  else
    POS_ARGS=(${POS_ARGS[*]} $1)
    shift 1
  fi
done

collect=${POS_ARGS[0]}
if [ "$collect" = "" ] ; then
  echo "Specify a sub-collection (1st positional arg): e.g., manner, compr"
  exit 1
fi

TEST_PART=${POS_ARGS[1]}
if [ "$TEST_PART" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (2d arg)"
  exit 1 
fi

CAND_PROV_TYPE=${POS_ARGS[2]}
if [ "$CAND_PROV_TYPE" = "" ] ; then
  echo "Specify a candidate type: lucene or nmslib (3d arg)"
  exit 1 
fi

EXPER_DIR_BASE=${POS_ARGS[3]}
if [ "$EXPER_DIR_BASE" = "" ] ; then
  echo "Specify a working directory (4th positional arg)!"
  exit 1
fi

EXTR_TYPE="${POS_ARGS[4]}"

if [ "$EXTR_TYPE" = "" ] ; then
  echo "Specify a feature extractor type (5th positional arg)"
  exit 1
fi


if [ "$EXTR_TYPE" != "none" ] ; then
  MODEL_FILE=${POS_ARGS[5]}

  if [ "$MODEL_FILE" = "" ] ; then
    echo "Specify a model file (6th positional) arg)!"
    exit 1
  fi
fi

NTEST_STR="${POS_ARGS[6]}"

if [ "$NTEST_STR" = "" ] ; then
  echo "Specify a comma-separated list of candidate records # retrieved for testing for each query (7th positional arg)!"
  exit 1
fi

NTEST_LIST=`echo $NTEST_STR|sed 's/,/ /g'`

EMBED_FILES="${POS_ARGS[7]}"

if [ "$EMBED_FILES" = "" ] ; then
  echo "Specify a (quoted) list of embedding files (8th positional arg)"
  exit 1
fi

HORDER_FILES="tran_embed.0,tran_embed.1,tran_embed.2,tran_embed.3,tran_embed.4"

# TODO is for Lucene, changes are coming for nmslib provider
if [ "$CAND_PROV_TYPE" = "lucene" -o "$CAND_PROV_TYPE" = "lucene_giza" ] ; then
  URI="lucene_index/$collect"
elif [ "$CAND_PROV_TYPE" = "nmslib" ] ; then
  if [ "$NMSLIB_ADDR" = "" ] ; then
    echo "Missing parameter -nmslib_addr"
    exit 1
  fi
  URI=$NMSLIB_ADDR
  if [ "$nmslib_fields_param" = "" ] ; then
    echo "Missing parameter -nmslib_fields"
    exit 1
  fi
else
  echo "Invalid provider type: $CAND_PROV_TYPE"
fi


EXPER_DIR="$EXPER_DIR_BASE/exper"
TREC_RUN_DIR="$EXPER_DIR_BASE/trec_runs"
REPORT_DIR="$EXPER_DIR_BASE/rep"

mkdir -p "$EXPER_DIR"
check "mkdir -p $EXPER_DIR"
mkdir -p $TREC_RUN_DIR
check "mkdir -p $TREC_RUN_DIR"
mkdir -p "$REPORT_DIR"
check "mkdir -p $REPORT_DIR"


echo "Deleting old reports from the directory: ${REPORT_DIR}"
rm -f ${REPORT_DIR}/*
check "rm -f ${REPORT_DIR}/*"

OUT_PREF_TEST="out_${collect}_${TEST_PART}_${EXTR_TYPE}"
FULL_OUT_PREF_TEST="$EXPER_DIR/$OUT_PREF_TEST"

query_log_file=${REPORT_DIR}/query.log
check "query_log_file=${REPORT_DIR}/query.log"

echo "Using $TEST_PART for testing!"
echo "Candidate provider type:        $CAND_PROV_TYPE"
echo "Candidate provider URI:         $URI"
echo "Experiment directory:           $EXPER_DIR"
echo "Report directory:               $REPORT_DIR"
if [ "$max_num_query_param" != "" ] ; then
  echo "Max number of queries param.:   $max_num_query_param"
fi
echo "Directory with TREC-style runs: $TREC_RUN_DIR"
echo "Model file:                     $MODEL_FILE"
if [ "$extr_type_interm_param" != "" ] ; then
  echo "Intermediate extractor param.:  $extr_type_interm_param"
fi
if [ "$model_interm_param" != "" ] ; then
  echo "Intermediate model param.:      $model_interm_param"
fi
if [ "$nmslib_fields_param" ] ; then
  echo "NMSLIB fields param:            $nmslib_fields_param"
fi

STAT_FILE="$REPORT_DIR/stat_file"

if [ "$EXTR_TYPE" = "none" ] ; then
  EXTR_FINAL_PARAM=""
else
  EXTR_FINAL_PARAM=" -model_final $MODEL_FILE -extr_type_final $EXTR_TYPE"
fi

scripts/query/run_query.sh  -u "$URI" $cand_qty_param $max_num_query_param $extr_type_interm_param $model_interm_param $nmslib_fields_param -q output/$collect/${TEST_PART}/SolrQuestionFile.txt  -n "$NTEST_STR" -o $TREC_RUN_DIR/run  -giza_root_dir tran/$collect/ -giza_iter_qty 5 -embed_dir WordEmbeddings/$collect  -embed_files  "$EMBED_FILES" -cand_prov $CAND_PROV_TYPE -memindex_dir memfwdindex/$collect -thread_qty $THREAD_QTY -horder_files "$HORDER_FILES" $maxQueryQtyTestParam -save_stat_file "$STAT_FILE" $EXTR_FINAL_PARAM $giza_expand_qty_param $giza_wght_expand_param  2>&1|tee $query_log_file
check_pipe "run_query.sh"

QRELS="output/$collect/${TEST_PART}/qrels.txt"


rm -f "${REPORT_DIR}/out_*"

for oneN in $NTEST_LIST ; do
  echo "======================================"
  echo "N=$oneN"
  echo "======================================"
  REPORT_PREF="${REPORT_DIR}/out_${oneN}"

  scripts/exper/eval_output.py trec_eval-9.0.4/trec_eval "$QRELS"  "${TREC_RUN_DIR}/run_${oneN}" "$REPORT_PREF" "$oneN"
  check "eval_output.py"
done

if [ "$delete_trec_runs" = "1" ] ; then
  echo "Deleting trec runs from the directory: ${TREC_RUN_DIR}"
  rm ${TREC_RUN_DIR}/*
  # There should be at least one run, so, if rm fails, it fails because files can't be deleted
  check "rm ${TREC_RUN_DIR}/*" 
else
  echo "Bzipping trec runs in the directory: ${TREC_RUN_DIR}"
  bzip2 ${TREC_RUN_DIR}/*
  # There should be at least one run, so, if rm fails, it fails because files can't be deleted
  check "bzip2 ${TREC_RUN_DIR}/*" 
fi

echo "Bzipping trec_eval output in the directory: ${REPORT_DIR}"
bzip2 ${REPORT_DIR}/*.trec_eval
check "bzip2 ${REPORT_DIR}/*.trec_eval"
