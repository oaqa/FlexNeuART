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

while [ $# -ne 0 ] ; do
  echo $1|grep "^-" >/dev/null 
  if [ $? = 0 ] ; then
    OPT_NAME="$1"
    OPT_VALUE="$2"
    OPT="$1 $2"
    if [ "$OPT_VALUE" = "" ] ; then  
      echo "Option $OPT_NAME requires an argument." >&2
      exit 1
    fi
    shift 2
    case $OPT_NAME in
      -thread_qty)
        THREAD_QTY=$OPT_VALUE 
        ;;
      -max_num_query)
        max_num_query_param=$OPT
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

# These flags are mostly for debug purposes
regen_feat="1"
recomp_model="1"

EXPER_DIR_BASE=${POS_ARGS[1]}
if [ "$EXPER_DIR_BASE" = "" ] ; then
  echo "Specify a working directory (2d positional arg)!"
  exit 1
fi

EXTR_TYPE="${POS_ARGS[2]}"

if [ "$EXTR_TYPE" = "" ] ; then
  echo "Specify a feature extractor type (3d positional arg)"
  exit 1
fi

if [ "$EXTR_TYPE" = "none" ] ; then
  "Exractor type cannot be 'none'!"
  exit 1
fi

N_TREES=${POS_ARGS[3]}
if [ "$N_TREES" = "" ] ; then
  echo "Specify the numbers of trees for LMART (4th positional arg)!"
  exit 1
fi

EMBED_FILES="${POS_ARGS[4]}"

if [ "$EMBED_FILES" = "" ] ; then
  echo "Specify a (quoted) list of embedding files (5th positional arg)"
  exit 1
fi

TRAIN_PART="${POS_ARGS[5]}"

if [ "$TRAIN_PART" = "" ] ; then
  echo "Specify a collection part for training, e.g., train (6th positional arg)"
  exit 1
fi

N_TRAIN="${POS_ARGS[6]}"

if [ "$N_TRAIN" = "" ] ; then
  echo "Specify a number of top-K records used for training, good values 15 for manner, 50 for compr (7th positional arg)"
  exit 1
fi


HORDER_FILES="tran_embed.0,tran_embed.1,tran_embed.2,tran_embed.3,tran_embed.4"

EXPER_DIR="$EXPER_DIR_BASE/exper"

mkdir -p "$EXPER_DIR"
check "mkdir -p $EXPER_DIR"

echo "Experiment directory:           $EXPER_DIR"

URI="lucene_index/$collect"

OUT_PREF_TRAIN="out_${collect}_${TRAIN_PART}_${EXTR_TYPE}"
FULL_OUT_PREF_TRAIN="$EXPER_DIR/$OUT_PREF_TRAIN"

if [ "$regen_feat" = "1" ] ; then

  scripts/query/gen_features.sh $collect $TRAIN_PART lucene $URI $N_TRAIN "$EXTR_TYPE" "$EXPER_DIR" $max_num_query_param  -out_pref "$OUT_PREF_TRAIN" -embed_files "$EMBED_FILES" -horder_files "$HORDER_FILES" -thread_qty $THREAD_QTY 2>&1
  check "scripts/query/gen_features.sh $collect $TRAIN_PART lucene $URI $N_TRAIN "$EXTR_TYPE" "$EXPER_DIR" $max_num_query_param  -out_pref "$OUT_PREF_TRAIN" -embed_files "$EMBED_FILES" -horder_files "$HORDER_FILES" -thread_qty $THREAD_QTY"
fi

if [ "$recomp_model" = "1" ] ; then
  MODEL_FILE="${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.model"

  model_log_file="$EXPER_DIR/model.log"
  echo > $model_log_file
    
  scripts/letor/ranklib_train_lmart.sh "${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.feat" "$MODEL_FILE" $N_TREES 2>&1 | tee -a "$model_log_file"
  check_pipe "scripts/letor/ranklib_train_lmart.sh "${FULL_OUT_PREF_TRAIN}_${N_TRAIN}.feat" "$MODEL_FILE" $N_TREES 2>&1 "
fi


