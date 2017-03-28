#/bin/bash

#PATH_TO_THE_SCRIPTS="${0%/*}"
#echo "A path to this script: $PATH_TO_THE_SCRIPTS"

. scripts/common.sh
. scripts/common_nmslib.sh

collect=${POS_ARGS[0]}
if [ "$collect" = "" ] ; then
  echo "Specify a collection: manner, compr (1st arg)"
  exit 1
fi

METHOD_NAME=${POS_ARGS[1]}
if [ "$METHOD_NAME" != "sw-graph" -a "$METHOD_NAME" != "napp" ] ; then
  echo "Specify a method name: napp or sw-graph (2d arg)"
  exit 1
fi

TEST_PART=${POS_ARGS[2]}
if [ "$TEST_PART" = "" ] ; then
  echo "Specify a test part, e.g., dev1 (3d arg)"
  exit 1 
fi

QREL_TYPE=${POS_ARGS[3]}
QREL_FILE=`get_qrel_file $QREL_TYPE "4th"`
check ""

if [ "$METHOD_NAME" = "napp" ] ; then
  HEAD_TYPE=${POS_ARGS[4]}
  if [ "$HEAD_TYPE" != "bm25_text" -a "$HEAD_TYPE" != "exper1" -a "$HEAD_TYPE" != "exper1_bm25" ] ; then
    echo "Specify a type of the header for $METHOD_NAME: bm25_text, exper1, or exper1_bm25 (5th argument)"
    exit 1
  fi
fi

NMSLIB_SPACE="qa1"
KNN_K=100
RBO_VALS="0.876404,0.937225,0.957932,0.968369,0.974663,0.978859,0.981865,0.984123,0.985893,0.987293"

if [ "$max_num_query" != "" ] ; then
  MAX_NUM_QUERY_PARAM=" --maxNumQuery $max_num_query "
fi

echo "Will be using $THREAD_QTY threads"
echo "Max # query parameter: $MAX_NUM_QUERY_PARAM"

NMSLIB_PREFIX="nmslib/$collect"

if [ "$METHOD_NAME" = "sw-graph" ] ; then
  source scripts/exper/params/swgraph_exper1_bm25_symm_text.sh
  check "source scripts/exper/params/swgraph_exper1_bm25_symm_text.sh"
else 
  if [ "$collect" = "compr" ] ; then
    NOP=x
  elif [ "$collect" = "stackoverflow" ] ; then
    NOP=x
  elif [ "$squad" = "squad" ] ; then
    NOP=x
  else
    echo "Unsupported collection: $collect for method $METHOD_NAME"
  fi
  param_file=scripts/exper/params/${METHOD_NAME}_${HEAD_TYPE}_${collect}.sh
  if [ ! -f "$param_file" ] ; then
    echo "Something is wrong, the parameter file: $param_file isn't found!"
    exit 1
  fi
  echo "Running a parameter file: $param_file"
  source $param_file
  check "source $param_file"
fi

QUERY_FILE="$NMSLIB_PREFIX/queries/$TEST_PART/${NMSLIB_FIELDS}_queries.txt"
INDEX_DIR="$NMSLIB_PREFIX/index/test/$NMSLIB_HEADER_NAME"

echo "Header file: $NMSLIB_HEADER_NAME"
echo "Query file: $QUERY_FILE"

if [ ! -f "$QUERY_FILE" ] ; then
  echo "Cannot find the query file: $QUERY_FILE"
  exit 1
fi

PREV_INDEX=""
PREV_INDEX_PARAM=""
PREV_QUERY_TIME_PARAMS=""

INDEX_NAME_ARR=()
INDEX_PARAM_ARR=()
QUERY_TIME_PARAM_ARR=()

pnum=$((${#PARAMS[*]}/2))
for ((i=0;i<$pnum;++i))
do
  ii=$((2*$i))
  iq=$((2*$i+1))
  in=$((2*$i+2))

  index_params=${PARAMS[$ii]}
  index_params_noslash=`echo $index_params|sed 's|/|_|g'`
  index_name=${INDEX_METHOD_PREFIX}_${index_params_noslash}
  query_time_params=${PARAMS[$iq]}

  echo "Index name: $index_name"
  echo "Query time parameters: $query_time_params"

  if [ "$PREV_INDEX" != "$index_name" ] ; then
    if [ "$PREV_INDEX" != "" ] ; then
      INDEX_NAME_ARR+=($PREV_INDEX)
      INDEX_PARAM_ARR+=($PREV_INDEX_PARAM)
      QUERY_TIME_PARAM_ARR+=($PREV_QUERY_TIME_PARAMS)
    fi
    PREV_QUERY_TIME_PARAMS=""
    PREV_INDEX=$index_name
    PREV_INDEX_PARAM=$index_params
  fi
  # Bash arrays cannot contain strings with spaces!
  PREV_QUERY_TIME_PARAMS+="_-t_${query_time_params}"
done
if [ "$PREV_INDEX" != "" ] ; then
  INDEX_NAME_ARR+=($PREV_INDEX)
  INDEX_PARAM_ARR+=($PREV_INDEX_PARAM)
  QUERY_TIME_PARAM_ARR+=("$PREV_QUERY_TIME_PARAMS")
fi

echo "================================="
echo "All index names:"
for t in ${INDEX_NAME_ARR[*]} ; do
  echo $t 
done
echo "================================="
echo "All index parameters:"
for t in ${INDEX_PARAM_ARR[*]} ; do
  echo $t
done
echo "================================="
echo "All query-time parameters:"
for t in ${QUERY_TIME_PARAM_ARR[*]} ; do
  echo $t
done
echo "================================="

qty=${#INDEX_NAME_ARR[*]}

for ((i=0;i<$qty;i++))
do
  QUERY_TIME_PARAMS=`echo ${QUERY_TIME_PARAM_ARR[$i]}|sed 's/_/ /g'`
  
  INDEX_NAME=$INDEX_DIR/${INDEX_NAME_ARR[$i]}

  echo "Index name: $INDEX_NAME" 
  echo "Query time parameters: $QUERY_TIME_PARAMS"

#  if [ ! -d "$INDEX_DIR" ] ; then
#    echo "$INDEX_DIR doesn't exist!"
#    exit 1
#  fi
#  if [ -f "$INDEX_NAME_COMP" ]
#  then
#    echo "Let's uncompress previously created index $INDEX_NAME_COMP"
#    gunzip "$INDEX_NAME_COMP"
#    check "gunzip $INDEX_NAME_COMP"
#  elif [ -f "$INDEX_NAME" ] ; then
#    echo "Found a previously created uncompressed index $INDEX_NAME"
#  else
#    echo "Cannot find a previously created index neither $INDEX_NAME nor $INDEX_NAME_COMP!"
#    exit 1
#  fi

#
  GS_CACHE_DIR="gs_cache/$collect/$NMSLIB_HEADER_NAME/$TEST_PART"
  REPORT_DIR="results/local/$collect/$TEST_PART/$INDEX_METHOD_PREFIX/$NMSLIB_HEADER_NAME"

  GS_CACHE_PREF="$GS_CACHE_DIR/${NMSLIB_SPACE}"

  if [ ! -d "$GS_CACHE_DIR" ] ; then
    mkdir -p "$GS_CACHE_DIR"
    check "mkdir -p "$GS_CACHE_DIR""
  fi

  if [ ! -d "$REPORT_DIR" ] ; then
    mkdir -p "$REPORT_DIR"
    check "mkdir -p "$REPORT_DIR""
  fi

  REPORT_PREF="${REPORT_DIR}/K=$KNN_K"

  if [ ! -d "$REPORT_PREF" ] ; then
    mkdir -p "$REPORT_PREF"
    check "mkdir -p "$REPORT_PREF""
  fi
  bash_cmd="../nmslib/similarity_search/release/experiment -s $NMSLIB_SPACE -g $GS_CACHE_PREF -i $NMSLIB_PREFIX/headers/$NMSLIB_HEADER_NAME \
                     --threadTestQty $THREAD_QTY \
                      -q "$QUERY_FILE" -k $KNN_K \
                      --pRBO $RBO_VALS
                      -m $NMSLIB_METHOD \
                      -L $INDEX_NAME \
                      $MAX_NUM_QUERY_PARAM \
                      $QUERY_TIME_PARAMS -o \"$REPORT_PREF/${INDEX_METHOD_PREFIX}\"   "
  echo "Command:"
  echo $bash_cmd
  #bash -c "$bash_cmd"
  check "$bash_cmd"

  #echo "Let's compress the index $INDEX_NAME"
  #gzip $INDEX_NAME
  #check "gzip $INDEX_NAME"
  #echo "Index is compressed!"

done

