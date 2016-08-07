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

NMSLIB_INDEX=$1

if [ "$NMSLIB_INDEX" = "" ] ; then
  echo "Specify a location of the sw-graph index (1st positional arg)!"
  exit 1
fi

COLLECT_NAME="compr"
#MAX_QUERY_QTY=10000
QUERY_SET="test"

SPACE="qa1"
K=500

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

echo "Index file: $NMSLIB_INDEX"
echo "Will be using $THREAD_QTY threads"

HEADERS=( header_avg_embed_word2vec_text_unlemm )
QUERIES=( text_unlemm_queries_sample0.25.txt )
QUERY_TIME_PARAM_ARR=( " -t initSearchAttempts=1,efSearch=450 -t initSearchAttempts=1,efSearch=900 -t initSearchAttempts=1,efSearch=1800 -t initSearchAttempts=1,efSearch=2100 -t initSearchAttempts=1,efSearch=2400 -t initSearchAttempts=1,efSearch=3600 -t initSearchAttempts=1,efSearch=7200" )

QTY=${#HEADERS[*]}

if [ ${#QUERIES[*]} != "$QTY" ] ; then
  echo "Number of QUERIES elements != # of HEADERS elements!"
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
  QUERY_TIME_PARAMS=${QUERY_TIME_PARAM_ARR[$i]}

  echo "Header: $HEADER_FILE full query path: $FULL_QUERY_PATH"

  GS_CACHE_DIR="gs_cache/$COLLECT_NAME/$HEADER_FILE"
  REPORT_DIR="results/testing/sw-graph/$COLLECT_NAME/$HEADER_FILE"

  REPORT_PREF="$REPORT_DIR/test_sw-graph_${QUERY_SET}"
  GS_CACHE_PREF="$GS_CACHE_DIR/${SPACE}_${QUERY_SET}"

  if [ ! -d "$GS_CACHE_DIR" ] ; then
    mkdir -p "$GS_CACHE_DIR"
    check "mkdir -p "$GS_CACHE_DIR""
  fi

  if [ ! -d "$REPORT_DIR" ] ; then
    mkdir -p "$REPORT_DIR"
    check "mkdir -p "$REPORT_DIR""
  fi

  # Let's not delete reports automatically!
  #rm -f $REPORT_PREF*

  bash_cmd="release/experiment -s $SPACE -g $GS_CACHE_PREF -i nmslib/$COLLECT_NAME/headers/$HEADER_FILE \
                       --threadTestQty $THREAD_QTY \
                        -q "$FULL_QUERY_PATH" -k $K \
                        -m sw-graph \
                        -L $NMSLIB_INDEX \
                        $QUERY_TIME_PARAMS -o $REPORT_PREF $ADD_FLAG  "

  # Next time we append to the report rather than overwrite it
  if [ "$ADD_FLAG" = "" ] ; then
    ADD_FLAG=" -a "
  fi
  echo "Command:"
  echo $bash_cmd
  bash -c "$bash_cmd"
  check "$bash_cmd"
done

