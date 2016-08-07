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

CHUNK_INDEX_SIZE=$((114*1024))
INDEX_PARAM_PREF="numPivot=8000,numPivotIndex=240"
HEADER_FILE="header_exper1"
MAX_QUERY_QTY=5000
QUERY_SET="dev1"
#FIELD_CODE="3field";
#FIELD_CODE_PIVOT="3field"
FIELD_CODE="text"
FIELD_CODE_PIVOT="text_field"
QUERY_FILE="${FIELD_CODE}_queries.txt"
GS_CACHE_DIR="gs_cache/$COLLECT_NAME/$HEADER_FILE"
REPORT_DIR="results/tunning/$COLLECT_NAME/$HEADER_FILE"
INDEX_DIR="indices/$COLLECT_NAME/$HEADER_FILE"
SPACE="qa1"
K=10

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

REPORT_PREF="$REPORT_DIR/tunning_napp_step1_${QUERY_SET}"
GS_CACHE_PREF="$GS_CACHE_DIR/${SPACE}_${QUERY_SET}"
INDEX_PREF="$INDEX_DIR/napp"

#Let's not delete reports automatically
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

echo "Chunk index size:     $CHUNK_INDEX_SIZE"
echo "Report file prefix:   $REPORT_PREF"  
echo "Gold standard prefix: $GS_CACHE_PREF"
echo "Index prefix:         $INDEX_PREF"
echo "Will be using $THREAD_QTY threads"

for pivot_term_qty in 300 1000 ; do
  for max_term_qty_K in 50 ; do
    pivot_file_name="pivots_${FIELD_CODE_PIVOT}_maxTermQty${max_term_qty_K}K_pivotTermQty${pivot_term_qty}"
    INDEX_PARAMS="${INDEX_PARAM_PREF},chunkIndexSize=$CHUNK_INDEX_SIZE,pivotFile=nmslib/$COLLECT_NAME/pivots/$pivot_file_name"

    INDEX_NAME="${INDEX_PREF}_${INDEX_PARAM_PREF}_${pivot_file_name}"

    echo "Index file name:      $INDEX_NAME"
    echo "Indexing parameters:  $INDEX_PARAMS"
    bash_cmd="release/experiment -s $SPACE -g $GS_CACHE_PREF -i nmslib/$COLLECT_NAME/headers/$HEADER_FILE \
                       --threadTestQty $THREAD_QTY \
                        -q nmslib/$COLLECT_NAME/queries/$QUERY_SET/${FIELD_CODE}_queries.txt -Q $MAX_QUERY_QTY -k $K \
                        -m napp_qa1 \
                        -c $INDEX_PARAMS -S $INDEX_NAME -L $INDEX_NAME \
                        -t numPivotSearch=15 -t numPivotSearch=16 -t numPivotSearch=17 -t numPivotSearch=18  -t numPivotSearch=19  -t numPivotSearch=20  -t numPivotSearch=21  -t numPivotSearch=22 -t numPivotSearch=23 \
                        -o $REPORT_PREF -a  "
    echo "Command:"
    echo $bash_cmd
    bash -c "$bash_cmd"
    check "$bash_cmd"


  done
done

