#/bin/bash

PATH_TO_THE_SCRIPTS="${0%/*}"
echo "A path to this script: $PATH_TO_THE_SCRIPTS"

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
COLLECT_NAME="compr"
#MAX_QUERY_QTY=10000
QUERY_SET="test"

SPACE="qa1"
CHUNK_INDEX_SIZE=$((114*1024))
K=100

GET_CPU_CORES_PATH="$PATH_TO_THE_SCRIPTS/../exper"
if [ ! -f "$GET_CPU_CORES_PATH/get_cpu_cores.py" ] ;then
  echo "Can't find the get_cpu_cores.py file in $GET_CPU_CORES_PATH"
  exit 1
fi
THREAD_QTY=`$GET_CPU_CORES_PATH/get_cpu_cores.py`
if [ "$THREAD_QTY" = "" ] ; then
  echo "Can't obtain the number of cores!"
  exit 1
fi

echo "Chunk index size: $CHUNK_INDEX_SIZE"
echo "Will be using $THREAD_QTY threads"

BEST_PIVOT_TERM_QTY=1000
BEST_MAX_TERM_QTY_K=50

HEADERS=(
         header_bm25_text \
         header_bm25_text \
         header_bm25_text \
         \
         header_exper1              \
         header_exper1              \
         header_exper1              \
         header_exper1              \
         header_exper1              \
         header_exper1              \
         header_exper1              \
        )
QUERIES=(
         text_queries_sample0.25.txt    \
         text_queries_sample0.25.txt    \
         text_queries_sample0.25.txt    \
         \
         text_queries_sample0.25.txt    \
         text_queries_sample0.25.txt    \
         text_queries_sample0.25.txt    \
         text_queries_sample0.25.txt    \
         text_queries_sample0.25.txt    \
         text_queries_sample0.25.txt    \
         text_queries_sample0.25.txt    \
        )
PIVOT_PREFS=(
            pivots_text_field \
            pivots_text_field \
            pivots_text_field \
            \
            pivots_text_field \
            pivots_text_field \
            pivots_text_field \
            pivots_text_field \
            pivots_text_field \
            pivots_text_field \
            pivots_text_field \
            )

NUM_PIVOT_INDEX_ARR=(
                    238 \
                    228 \
                    200 \
                    \
                     320    \
                     300    \
                     250    \
                     220    \
                     228    \
                     200    \
                     210    \
            )

QUERY_TIME_PARAM_ARR=(
      "-t numPivotSearch=12" \
      "-t numPivotSearch=13" \
      "-t numPivotSearch=11 -t numPivotSearch=12 -t numPivotSearch=13 -t numPivotSearch=14 -t numPivotSearch=15 -t numPivotSearch=17 -t numPivotSearch=19 " \
         \
      "-t numPivotSearch=22" \
      "-t numPivotSearch=22" \
      "-t numPivotSearch=17" \
      "-t numPivotSearch=14" \
      "-t numPivotSearch=16" \
      "-t numPivotSearch=13 -t numPivotSearch=14 -t numPivotSearch=15 -t numPivotSearch=18 " \
      "-t numPivotSearch=17" \
            )

QTY=${#HEADERS[*]}

if [ ${#QUERIES[*]} != "$QTY" ] ; then
  echo "Number of QUERIES elements != # of HEADERS elements!"
  exit 1
fi
if [ ${#PIVOT_PREFS[*]} != "$QTY" ] ; then
  echo "Number of PIVOT_PREFS elements != # of HEADERS elements!"
  exit 1
fi
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
  QUERY_FILE=${QUERIES[$i]}
  FULL_QUERY_PATH="nmslib/$COLLECT_NAME/queries/$QUERY_SET/$QUERY_FILE"
  NUM_PIVOT_INDEX=${NUM_PIVOT_INDEX_ARR[$i]}
  QUERY_TIME_PARAMS=${QUERY_TIME_PARAM_ARR[$i]}

  echo "Header: $HEADER_FILE full query path: $FULL_QUERY_PATH"

  GS_CACHE_DIR="gs_cache/$COLLECT_NAME/$HEADER_FILE"
  REPORT_DIR="results/testing/napp/$COLLECT_NAME/$HEADER_FILE"
  INDEX_DIR="indices/$COLLECT_NAME/$HEADER_FILE"

  REPORT_PREF="$REPORT_DIR/test_napp_${QUERY_SET}"
  GS_CACHE_PREF="$GS_CACHE_DIR/${SPACE}_${QUERY_SET}"
  INDEX_PREF="$INDEX_DIR/napp"

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

  # Let's not delete reports automatically!
  #rm -f $REPORT_PREF*

  PIVOT_FILE_NAME="${PIVOT_PREFS[$i]}_maxTermQty${BEST_MAX_TERM_QTY_K}K_pivotTermQty${BEST_PIVOT_TERM_QTY}"
  PIVOT_FILE="nmslib/$COLLECT_NAME/pivots/$PIVOT_FILE_NAME"
  if [ ! -f "$PIVOT_FILE" ] ; then
    echo "Cannot find the pivot file: $PIVOT_FILE"
    exit 1
  fi
  INDEX_PARAMS="chunkIndexSize=$CHUNK_INDEX_SIZE,numPivot=$NUM_PIVOT,numPivotIndex=$NUM_PIVOT_INDEX,pivotFile=$PIVOT_FILE"

  INDEX_NAME="${INDEX_PREF}_numPivot=$NUM_PIVOT,numPivotIndex=${NUM_PIVOT_INDEX}_${PIVOT_FILE_NAME}"

  INDEX_NAME_COMP="${INDEX_NAME}.gz"
  if [ -f "$INDEX_NAME_COMP" ]
  then
    echo "Let's uncompress previously created index"
    gunzip "$INDEX_NAME_COMP"
    check "gunzip $INDEX_NAME_COMP"
  elif [ -f "$INDEX_NAME" ] ; then
    echo "Found a previously created uncompressed index"
  else
    echo "Cannot find a previously created index!"
    exit 1
  fi

  bash_cmd="release/experiment -s $SPACE -g $GS_CACHE_PREF -i nmslib/$COLLECT_NAME/headers/$HEADER_FILE \
                       --threadTestQty $THREAD_QTY \
                        -q "$FULL_QUERY_PATH" -k $K \
                        -m napp_qa1 \
                        -L $INDEX_NAME \
                        $QUERY_TIME_PARAMS -o $REPORT_PREF $ADD_FLAG  "

  # Next time we append to the report rather than overwrite it
  if [ "$ADD_FLAG" = "" ] ; then
    ADD_FLAG=" -a "
  fi
  echo "Command:"
  echo $bash_cmd
  bash -c "$bash_cmd"
  check "$bash_cmd"

  #echo "Let's compress the index $INDEX_NAME"
  #gzip $INDEX_NAME
  #check "gzip $INDEX_NAME"
  #echo "Index is compressed!"

done

