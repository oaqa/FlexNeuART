#/bin/bash

#PATH_TO_THE_SCRIPTS="${0%/*}"
#echo "A path to this script: $PATH_TO_THE_SCRIPTS"

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

NUM_PIVOT=8000

COLLECT=$1

if [ "$COLLECT" = "" ] ; then
  echo "Specify the collection name: compr or stackoverflow (1st arg)"
  exit 1
fi

#MAX_QUERY_QTY=10000
QUERY_SET="test"

SPACE="qa1"
#K="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100"
# Should be in the decreasing order
K_ARR=(100 10)

THREAD_QTY=`scripts/exper/get_cpu_cores.py`
if [ "$THREAD_QTY" = "" ] ; then
  echo "Can't obtain the number of cores!"
  exit 1
fi

echo "Will be using $THREAD_QTY threads"

BEST_PIVOT_TERM_QTY=1000
BEST_MAX_TERM_QTY_K=50

NMSLIB_PREFIX="nmslib/$COLLECT"


PIVOT_FILE_NAME="${PIVOT_PREFS[$i]}_maxTermQty${BEST_MAX_TERM_QTY_K}K_pivotTermQty${BEST_PIVOT_TERM_QTY}"
PIVOT_FILE="$NMSLIB_PREFIX/pivots/pivots_text_field_maxTermQty${BEST_MAX_TERM_QTY_K}K_pivotTermQty${BEST_PIVOT_TERM_QTY}"

if [ ! -f "$PIVOT_FILE" ] ; then
  echo "Cannot find the pivot file: $PIVOT_FILE"
  exit 1
fi
PIVOT_FILE_PARAM="pivotFile=$PIVOT_FILE"

QUERY_FILE="$NMSLIB_PREFIX/queries/$QUERY_SET/text_queries.txt"

if [ ! -f "$QUERY_FILE" ] ; then
  echo "Cannot find the query file: $QUERY_FILE"
  exit 1
fi

if [ "$COLLECT" = "compr" ] ; then
  echo "Using parameters tuned for compr!"
  HEADERS=(
         header_bm25_text \
         header_bm25_text \
         header_bm25_text \
         \
         header_exper1_hash_payload              \
         header_exper1_hash_payload              \
         header_exper1_hash_payload              \
         header_exper1_hash_payload              \
         header_exper1_hash_payload              \
        )
  NUM_PIVOT_INDEX_ARR=(
                    238 \
                    228 \
                    200 \
                    \
                     300    \
                     250    \
                     200    \
                     150    \
                     100    \
            )
  QUERY_TIME_PARAM_ARR=(
      "-t numPivotSearch=12" \
      "-t numPivotSearch=13" \
      "-t numPivotSearch=11 -t numPivotSearch=12 -t numPivotSearch=13 -t numPivotSearch=14 -t numPivotSearch=15 -t numPivotSearch=17 -t numPivotSearch=19 " \
         \
      "-t numPivotSearch=18 -t numPivotSearch=22 " \
      "-t numPivotSearch=17 -t numPivotSearch=19 " \
      "-t numPivotSearch=13 -t numPivotSearch=15 -t numPivotSearch=19 " \
      "-t numPivotSearch=9" \
      "-t numPivotSearch=7" \
            )
elif [ "$COLLECT" = "stackoverflow" ] ; then
  echo "Using parameters tuned for stackoverflow!"
  HEADERS=(
         header_bm25_text \
         header_bm25_text \
         \
         header_exper1_hash_payload              \
         header_exper1_hash_payload              \
         header_exper1_hash_payload              \
         header_exper1_hash_payload              \
        )
  NUM_PIVOT_INDEX_ARR=(
                    200 \
                    100 \
                    \
                     250    \
                     200    \
                     150    \
                     100    \
            )
  QUERY_TIME_PARAM_ARR=(
      "-t numPivotSearch=13 -t numPivotSearch=14 " \
      "-t numPivotSearch=5 -t numPivotSearch=6 -t numPivotSearch=7 -t numPivotSearch=8 -t numPivotSearch=9 -t numPivotSearch=10 -t numPivotSearch=11  -t numPivotSearch=12 " \
         \
      "-t numPivotSearch=16 -t numPivotSearch=17 -t numPivotSearch=18 -t numPivotSearch=19 -t numPivotSearch=20 " \
      "-t numPivotSearch=14 -t numPivotSearch=16 -t numPivotSearch=18 " \
      "-t numPivotSearch=11" \
      "-t numPivotSearch=7" \
            )
else
  echo "Unsupported collection: $COLLECT"
fi

QTY=${#HEADERS[*]}

if [ ${#NUM_PIVOT_INDEX_ARR[*]} != "$QTY" ] ; then
  echo "Number of NUM_PIVOT_INDEX_ARR elements != # of HEADERS elements!"
  exit 1
fi
if [ ${#QUERY_TIME_PARAM_ARR[*]} != "$QTY" ] ; then
  echo "Number of QUERY_TIME_PARAM_ARR elements != # of HEADERS elements!"
  exit 1
fi

ADD_FLAG=""
PREV_HEADER=""

for ((i=0;i<$QTY;i++))
do
  HEADER_FILE=${HEADERS[$i]}
  if [ "$PREV_HEADER" != "$HEADER_FILE" ] ; then
    ADD_FLAG=""
  fi
  PREV_HEADER=$HEADER_FILE
  NUM_PIVOT_INDEX=${NUM_PIVOT_INDEX_ARR[$i]}
  QUERY_TIME_PARAMS=${QUERY_TIME_PARAM_ARR[$i]}

  echo "Header: $HEADER_FILE query file: $QUERY_FILE"

  GS_CACHE_DIR="gs_cache/$COLLECT/$HEADER_FILE/$QUERY_SET"
  REPORT_DIR="results/local/$COLLECT/$QUERY_SET/napp/$HEADER_FILE"
  INDEX_DIR="$NMSLIB_PREFIX/index/$HEADER_FILE"

  GS_CACHE_PREF="$GS_CACHE_DIR/${SPACE}"

  if [ ! -d "$GS_CACHE_DIR" ] ; then
    mkdir -p "$GS_CACHE_DIR"
    check "mkdir -p "$GS_CACHE_DIR""
  fi

  if [ ! -d "$REPORT_DIR" ] ; then
    mkdir -p "$REPORT_DIR"
    check "mkdir -p "$REPORT_DIR""
  fi

  if [ ! -d "$INDEX_DIR" ] ; then
    echo "$INDEX_DIR doesn't exist!"
    exit 1
  fi

  INDEX_PARAMS="numPivot=$NUM_PIVOT,numPivotIndex=$NUM_PIVOT_INDEX,$PIVOT_FILE_PARAM"
  INDEX_PARAM_NO_SLASH=`echo $INDEX_PARAMS|sed 's|/|_|g'`
  INDEX_NAME=$INDEX_DIR/napp_${INDEX_PARAM_NO_SLASH}
  INDEX_NAME_COMP="${INDEX_NAME}.gz"
  if [ -f "$INDEX_NAME_COMP" ]
  then
    echo "Let's uncompress previously created index $INDEX_NAME_COMP"
    gunzip "$INDEX_NAME_COMP"
    check "gunzip $INDEX_NAME_COMP"
  elif [ -f "$INDEX_NAME" ] ; then
    echo "Found a previously created uncompressed index $INDEX_NAME"
  else
    echo "Cannot find a previously created index neither $INDEX_NAME nor $INDEX_NAME_COMP!"
    exit 1
  fi

  # Values in $K_ARR should be in the decreasing order
  for k in ${K_ARR[*]} ; do
    REPORT_PREF="$REPORT_DIR/K=$k/"

    if [ ! -d "$REPORT_PREF" ] ; then
      mkdir -p "$REPORT_PREF"
      check "mkdir -p "$REPORT_PREF""
    fi

    bash_cmd="../nmslib/similarity_search/release/experiment -s $SPACE -g $GS_CACHE_PREF -i $NMSLIB_PREFIX/headers/$HEADER_FILE \
                       --threadTestQty $THREAD_QTY \
                        -q "$QUERY_FILE" -k $k \
                        -m napp_qa1 \
                        -L $INDEX_NAME \
                        $QUERY_TIME_PARAMS -o "$REPORT_PREF/napp" $ADD_FLAG  "
    echo "Command:"
    echo $bash_cmd
    bash -c "$bash_cmd"
    check "$bash_cmd"
  done

  # Next time we append to the report rather than overwrite it
  if [ "$ADD_FLAG" = "" ] ; then
    ADD_FLAG=" -a "
  fi

  echo "Let's compress the index $INDEX_NAME"
  gzip $INDEX_NAME
  check "gzip $INDEX_NAME"
  echo "Index is compressed!"

done

