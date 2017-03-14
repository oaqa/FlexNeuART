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

COLLECT_NAME=$1

if [ "$COLLECT_NAME" = "" ] ; then
  echo "Specify the collection name, e.g., compr, stackoverflow (1st arg)"
  exit 1
fi

USE_ALTERN_PIVOT_DIST=$2

if [ "$USE_ALTERN_PIVOT_DIST" = "" ] ; then
  echo "Specify a 'use alternate pivot distance flag'"
fi

NUM_PIVOT_INDEX=$3

if [ "$NUM_PIVOT_INDEX" = "" ] ; then
  echo "Specify the list of numbers of pivots to index (3d arg)"
  exit 1
fi

NUM_PIVOT_SEARCH_ARG=$4

if [ "$NUM_PIVOT_SEARCH_ARG" = "" ] ; then
  echo "Specify the list of numbers of pivots to search (4th arg)"
  exit 1
fi
NUM_PIVOT_SEARCH=`for t in $NUM_PIVOT_SEARCH_ARG ; do echo -n " -t numPivotSearch=$t" ; done`
echo "NUM_PIVOT_SEARCH=$NUM_PIVOT_SEARCH"

NUM_PIVOT=8000
HEADER_FILE_BASE="header_exper1_hash_payload"
HEADER_FILE_ALTERN="header_exper1_bm25_hash_payload"

if [ "$USE_ALTERN_PIVOT_DIST" = "1" ] ; then
  HEADER_FILE=$HEADER_FILE_ALTERN
else
  HEADER_FILE=$HEADER_FILE_BASE
fi

MAX_QUERY_QTY=5000
if [ "$5" != "" ] ; then
  MAX_QUERY_QTY="$5"
fi

QUERY_SET="dev1"

if [ "$COLLECT_NAME" = "squad" ] ; then
  FIELD_CODE_PIVOT="2field"
  FIELD_CODE_QUERY="text,text_alias1"
else
  FIELD_CODE_PIVOT="text_field"
  FIELD_CODE_QUERY="text"
fi

QUERY_FILE_NAME="${FIELD_CODE_QUERY}_queries.txt"
QUERY_FILE="nmslib/$COLLECT_NAME/queries/$QUERY_SET/${QUERY_FILE_NAME}"

qty=`wc -l $QUERY_FILE`
check "qty=`wc -l $QUERY_FILE`"
if [ "$qty" -lt "$MAX_QUERY_QTY" ] ; then
  echo "Reducing the maximum # of queries, b/c the actual test set is smaller."
  MAX_QUERY_QTY=$qty
fi
GS_CACHE_DIR="gs_cache/$COLLECT_NAME/$HEADER_FILE"
REPORT_DIR="results/tunning/$COLLECT_NAME/$HEADER_FILE"
INDEX_DIR="nmslib/$COLLECT_NAME/index/tuning/$HEADER_FILE"
SPACE="qa1"
#CHUNK_INDEX_SIZE=$((114*1024))
K=10

echo "Header file:  $HEADER_FILE"
echo "Report dir:   $REPORT_DIR"
echo "Index dir:    $INDEX_DIR"
echo "GS cache dir: $GS_CACHE_DIR"
echo "Max. query #: $MAX_QUERY_QTY"

BEST_PIVOT_TERM_QTY=1000
BEST_MAX_TERM_QTY_K=50

if [ ! -d "$GS_CACHE_DIR" ] ; then
  mkdir -p "$GS_CACHE_DIR"
  check "mkdir -p "$GS_CACHE_DIR""
fi

if [ ! -d "$REPORT_DIR" ] ; then
  mkdir -p "$REPORT_DIR"
  check "mkdir -p "$REPORT_DIR""
fi

if [ ! -d "$INDEX_DIR" ] ; then
  mkdir -p "$INDEX_DIR"
  check "mkdir -p "$INDEX_DIR""
fi

REPORT_PREF="$REPORT_DIR/tunning_napp_${QUERY_SET}"
GS_CACHE_PREF="$GS_CACHE_DIR/${SPACE}_${QUERY_SET}"
INDEX_PREF="$INDEX_DIR/napp"

# Let's not delete reports automatically!
#rm -f $REPORT_PREF*

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

#echo "Chunk index size: $CHUNK_INDEX_SIZE"
echo "Will be using $THREAD_QTY threads"

#for((numPivotIndex=220;numPivotIndex<261;numPivotIndex+=2))
#for numPivotIndex in 200 210 220 222 224 226 228 230 232 234 236 238 240 242 244 246 248 250 252 254 270 280 300 320
#for numPivotIndex in 250 275 300 325 350
for numPivotIndex in $NUM_PIVOT_INDEX 
do
  pivot_file_name="pivots_${FIELD_CODE_PIVOT}_maxTermQty${BEST_MAX_TERM_QTY_K}K_pivotTermQty${BEST_PIVOT_TERM_QTY}"
  #INDEX_PARAMS="chunkIndexSize=$CHUNK_INDEX_SIZE,numPivot=$NUM_PIVOT,numPivotIndex=$numPivotIndex,pivotFile=nmslib/$COLLECT_NAME/pivots/$pivot_file_name"
  INDEX_PARAMS="numPivot=$NUM_PIVOT,numPivotIndex=$numPivotIndex,pivotFile=nmslib/$COLLECT_NAME/pivots/$pivot_file_name"

  INDEX_PARAMS_NOSLASH=`echo $INDEX_PARAMS|sed 's|/|_|g'`
  INDEX_NAME="${INDEX_PREF}_${INDEX_PARAMS_NOSLASH}"

  INDEX_NAME_COMP="${INDEX_NAME}.gz"
  if [ -f "$INDEX_NAME_COMP" ]
  then
    echo "Let's uncompress previously created index"
    gunzip "$INDEX_NAME_COMP"
    check "gunzip $INDEX_NAME_COMP"
  fi

  # The commented-out settings are not-so-bad settings for COMPR
  #if [ $numPivotIndex -lt 260 ] ; then
    #NUM_PIVOT_SEARCH="-t numPivotSearch=10 -t numPivotSearch=11 -t numPivotSearch=12 -t numPivotSearch=13  -t numPivotSearch=14  -t numPivotSearch=15  -t numPivotSearch=16  -t numPivotSearch=17 -t numPivotSearch=18  -t numPivotSearch=19  -t numPivotSearch=20  "
  #elif [ $numPivotIndex -lt 280 ]
  #then
    #NUM_PIVOT_SEARCH="-t numPivotSearch=13 -t numPivotSearch=14 -t numPivotSearch=15 -t numPivotSearch=16 -t numPivotSearch=17 -t numPivotSearch=18  -t numPivotSearch=19  -t numPivotSearch=20  -t numPivotSearch=21  -t numPivotSearch=22 -t numPivotSearch=23  "
  #elif [ $numPivotIndex -lt 310 ]
  #then
    #NUM_PIVOT_SEARCH=" -t numPivotSearch=16 -t numPivotSearch=17 -t numPivotSearch=18  -t numPivotSearch=19  -t numPivotSearch=20 -t numPivotSearch=21  -t numPivotSearch=22 -t numPivotSearch=23  -t numPivotSearch=24  -t numPivotSearch=25 -t numPivotSearch=26  "
  #elif [ $numPivotIndex -lt 330 ]
  #then
    #NUM_PIVOT_SEARCH=" -t numPivotSearch=19  -t numPivotSearch=20 -t numPivotSearch=21  -t numPivotSearch=22 -t numPivotSearch=23  -t numPivotSearch=24   -t numPivotSearch=25 -t numPivotSearch=26 -t numPivotSearch=27 -t numPivotSearch=28 -t numPivotSearch=29 "
  #else
    #NUM_PIVOT_SEARCH=" -t numPivotSearch=22 -t numPivotSearch=23  -t numPivotSearch=24  -t numPivotSearch=25 -t numPivotSearch=26 -t numPivotSearch=27 -t numPivotSearch=28 -t numPivotSearch=29  -t numPivotSearch=30 -t numPivotSearch=31 -t numPivotSearch=32 "
  #fi 

  bash_cmd="../nmslib/similarity_search/release/experiment -s $SPACE -g $GS_CACHE_PREF -i nmslib/$COLLECT_NAME/headers/$HEADER_FILE \
                     --threadTestQty $THREAD_QTY \
                      -q ${QUERY_FILE} -Q $MAX_QUERY_QTY -k $K \
                      -m napp_qa1 \
                      -c $INDEX_PARAMS -S $INDEX_NAME -L $INDEX_NAME \
                      $NUM_PIVOT_SEARCH -o $REPORT_PREF -a  "
  echo "Command:"
  echo $bash_cmd
  bash -c "$bash_cmd"
  check "$bash_cmd"

  echo "Let's compress the index"
  gzip $INDEX_NAME
  check "gzip $INDEX_NAME"
  echo "Index is compressed!"

done

