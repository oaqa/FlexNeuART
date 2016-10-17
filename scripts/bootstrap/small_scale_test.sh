#!/bin/bash

INSTALL_DIR=$1

if [ "$INSTALL_DIR" = "" ] ; then
  echo "Specify an installation directory (1st arg)"
  exit 1
fi

if [ ! -d "$INSTALL_DIR" ] ; then
  echo "Not a directory: $INSTALL_DIR"
  exit 1
fi

DATA_DIR="$2"

if [ "$DATA_DIR" = "" ] ; then
  echo "Specify a data directory (2d arg)"
  exit 1
fi

if [ ! -d "$DATA_DIR" ] ; then
  echo "Not a directory: $DATA_DIR"
  exit 1
fi

RAW_DIR="$3"

if [ "$RAW_DIR" = "" ] ; then
  echo "Specify a directory with raw input files (3d arg)"
  exit 1
fi

if [ ! -d "$RAW_DIR" ] ; then
  echo "Not a directory: $RAW_DIR"
  exit 1
fi

# Let's check if you downloaded GoogleNews embeddings
EmbedName="GoogleNews-vectors-negative300.bin"
EmbedDir="$DATA_DIR/Embeddings/Complete/"
if [ ! -f "$EmbedDir/$EmbedName" ] ; then
  echo "Download the file GoogleNews-vectors-negative300.bin and place it $EmbedDir (from https://code.google.com/archive/p/word2vec/)" 
  exit 1
fi

KNN4QA_DIR="$INSTALL_DIR/knn4qa"

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

function run_cmd {
  cmd="$1"
  bash -c "$cmd"
  check "$cmd"
}

cd $KNN4QA_DIR ; check "cd $KNN4QA_DIR"

CPE_DIR="$KNN4QA_DIR/src/main/resources/descriptors/collection_processing_engines"

# This function exits while trying to restore UIMA descriptor files
function restore_desc_exit {
  exit_status=$1
  echo "Let's try to restore descriptor files in $CPE_DIR"
  cc $CPE_DIR ; check "cd $CPE_DIR"
  git checkout * ; check "git checkout *"
  exit $exit_status
}

# 1. Splitting collections

function split_collection {
  col=$1
  inpf=$2
  outf=$3
  probs=$4
  subsets=$t
  if [ ! -f "$RAW_DIR/$inpf" ] ; then
    echo "Cannot find input raw file $RAW_DIR/$inpf for collection $col"
    exit 1
  fi

  CMD_NAME="splitting collection $col"
  LOG_FILE="log_split.$col"
  echo "$CMD_NAME logging to $LOG_FILE"
  scripts/data/split_input.sh -i "$RAW_DIR/inpf"  -o "$DATA_DIR/input/$col/$outf -p "$probs" -n "$subsets" >$LOG_FILE
  check "$CMD_NAME$
}

# 2. Annotating collections
split_collection "manner" "manner.xml.bz2" "manner-v2.0" "0.60,0.05,0.15,0.20" "train,dev1,dev2,test"
split_collection "compr" "FullOct2007.xml.bz2" "comprehensive-Oct2007" "0.048,0.0024,0.0094,0.0117,0.9285" "train,dev1,dev2,test,tran"
split_collection "stackoverflow" "PostsNoCode2016-04-28.xml.bz2"  "StackOverflow" "0.048,0.0024,0.0094,0.0117,0.9285" "train,dev1,dev2,test,tran"

echo "Splitting collections into subsets is completed!"

# 2.1 We want smaller subsets, so we replace CPE descriptors with descriptors that limit the number of documents processed

echo "cp $KNN4QA_DIR/scripts/bootstrap/collection_processing_engines.quick/*.xml "

echo "Replaced existing CPE descriptors with descriptors for collection samples"
echo "IMPORTANT NOTE!!!! If something goes awfully wrong you may need to manually 'git checkout' in $CPE_DIR"

# 2.2 An actual annotation process

for col in manner compr ComprMinusManner stackoverflow ; do
  CMD_NAME="annotation pipeline for $col"
  LOG_FILE="log_annot.$col"
  echo "Running $CMD_NAME logging to $LOG_FILE"
  scripts/uima/run_annot_pipeline.sh manner >$LOG_FILE
  check "$CMD_NAME logging to $LOG_FILE"
done

echo "Annotation process is completed!"

# 3 Creating indices

for col in manner compr stackoverflow ; do
  CMD_NAME="creating Lucene index for $col"
  LOG_FILE="log_lucene_index.$col"
  echo "Running $CMD_NAME logging to $LOG_FILE"
  scripts/index/create_lucene_index.sh $col > $LOG_FILE
  check "$CMD_NAME logging to $LOG_FILE"

  CMD_NAME="creating forward index for $col"
  LOG_FILE="log_inmemfwd_index.$col"
  echo "Running $CMD_NAME logging to $LOG_FILE"
  scripts/index/create_inmemfwd_index.sh $col > $LOG_FILE
  check "$CMD_NAME logging to $LOG_FILE"
done

echo "All major work is done!"
restore_desc_exit 0

