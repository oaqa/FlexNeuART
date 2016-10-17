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

function check {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    restore_desc_exit 1
  fi
}

function check_pipe {
  f="${PIPESTATUS[*]}"
  name=$1
  if [ "$f" != "0 0" ] ; then
    echo "******************************************"
    echo "* Failed (pipe): $name, exit statuses: $f "
    echo "******************************************"
    restore_desc_exit 1
  fi
}

function run_cmd {
  cmd="$1"
  bash -c "$cmd"
  check "$cmd"
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

echo "Lucene and forward indices are created!"

# 4 Creating IBM Model 1 translation files
for col in ComprMinusManner compr stackoverflow ; do
  for field in text text_unlemm ; do
    CMD_NAME="Building IBM Model 1 for $col field $field"
    LOG_FILE="log_model1.${col}_${field}"
    echo "Running $CMD_NAME logging to $LOG_FILE"
    scripts/giza/create_tran.sh output tran $col tran $field /home/ubuntu/soft/giza-pp 0 5  > $LOG_FILE
    check "$CMD_NAME logging to $LOG_FILE"
  done
done

echo "IBM Model 1 translation files are created!"

# 5 Building derivative data
for col in ComprMinusManner compr stackoverflow ; do
  CMD_NAME="Generative derivative data for $col"
  LOG_FILE="log_deriv.$col"
  echo "Running $CMD_NAME logging to $LOG_FILE"
  scripts/data/gen_derivative_data.sh $col
  check "$CMD_NAME logging to $LOG_FILE"
done

echo "Derivative data is generated!"

# 6 Let's generate pivots and queries
for col in compr stackoverflow ; do
  CMD_NAME="generative pivots for $col"
  LOG_FILE="log_pivots.$col"
  echo "Running $CMD_NAME logging to $LOG_FILE"
  scripts/nmslib/gen_nmslib_pivots.sh /home/ubuntu/data $col
  check "$CMD_NAME logging to $LOG_FILE"
done
echo "Pivots are generated!"

for col in compr stackoverflow ; do
  CMD_NAME="generative queries for $col"
  LOG_FILE="log_queries.$col"
  echo "Running $CMD_NAME logging to $LOG_FILE"
  scripts/nmslib/gen_nmslib_queries.sh /home/ubuntu/data $col
  check "$CMD_NAME logging to $LOG_FILE"
done
echo "Queries are generated!"

# 7 Feature experiments 
for col in ComprMinusManner compr stackoverflow ; do
  CMD_NAME="running feature experiments for $col (using 2 parallel processes)"
  LOG_FILE="log_feature_exper.$col"
  echo "Running $CMD_NAME logging to $LOG_FILE"
  scripts/exper/run_feature_experiments.sh manner scripts/exper/feature_desc/test4combinations.txt 2 2>&1| tee $LOG_FILE
  check_pipe "$CMD_NAME logging to $LOG_FILE"
done
echo "Test feature experiments are finished!"

# 8 Main experiments

for col in compr stackoverflow ; do
  # First all brute-force
  TEST_SUBSET="test"
  run_cmd "scripts/exper/test_nmslib_server_bruteforce_bm25_text.sh $col $TEST_SUBSET"
  run_cmd "scripts/exper/test_nmslib_server_bruteforce_cosine_text.sh $col $TEST_SUBSET"
  run_cmd "scripts/exper/test_nmslib_server_bruteforce_exper1.sh $col $TEST_SUBSET"
  run_cmd "scripts/exper/test_nmslib_server_bruteforce_swgraph.sh $col $TEST_SUBSET"

  # Now tests with indices
  run_cmd "scripts/exper/test_nmslib_server_napp_bm25_text_${col}.sh $TEST_SUBSET"
  run_cmd "scripts/exper/test_nmslib_server_napp_exper1_${col}.sh $TEST_SUBSET"
  run_cmd "scripts/exper/test_nmslib_server_swgraph.sh $col $TEST_SUBSET"
done


echo "Main experiments are finished!"
echo "All major work is done!"
restore_desc_exit 0

