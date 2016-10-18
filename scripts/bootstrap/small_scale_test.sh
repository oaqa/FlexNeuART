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

function check_simple {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    exit 1
  fi
}

cd $INSTALL_DIR ; check_simple "cd $INSTALL_DIR" ; INSTALL_DIR=$PWD ; cd - ; check_simple "cd -"
cd $DATA_DIR ; check_simple "cd $DATA_DIR" ; DATA_DIR=$PWD ; cd - ; check_simple "cd -"
cd $RAW_DIR ; check_simple "cd $RAW_DIR" ; RAW_DIR=$PWD ; cd - ; check_simple "cd -"

KNN4QA_DIR="$INSTALL_DIR/knn4qa"

cd $KNN4QA_DIR ; check_simple "cd $KNN4QA_DIR"

CPE_DIR="$KNN4QA_DIR/src/main/resources/descriptors/collection_processing_engines/"

echo "Full installation path:       $INSTALL_DIR"
echo "Full data path:               $DATA_DIR"
echo "Full path to raw input files: $RAW_DIR"
echo "Directory with Uima CPE files:$CPE_DIR"

# Let's check if you downloaded GoogleNews embeddings
EmbedName="GoogleNews-vectors-negative300.bin"
EmbedDir="$DATA_DIR/WordEmbeddings/Complete/"
if [ ! -f "$EmbedDir/$EmbedName" ] ; then
  echo "Download the file GoogleNews-vectors-negative300.bin and place it $EmbedDir (from https://code.google.com/archive/p/word2vec/)" 
  exit 1
fi

if [ ! -d "$CPE_DIR" ] ; then
  echo "Not a directory: $CPE_DIR"
  exit 1 
fi

# This function exits while trying to restore UIMA descriptor files
# It should use only check_simple or else there will be an infinite recursion!
function restore_desc_exit {
  exit_status=$1
  echo "Let's try to restore descriptor files in $CPE_DIR"
  cd $CPE_DIR ; check_simple "cd $CPE_DIR"
  git checkout * ; check_simple "git checkout *"
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

function run_cmd_complex {
  CMD="$1"
  CMD_NAME="$2"
  LOG_FILE="$3"
  echo "$CMD_NAME logging to $LOG_FILE"
  echo "Command line:"
  echo "$CMD"
  bash -c "$CMD" &>$LOG_FILE
  check "$CMD_NAME"
}

# 1. Splitting collections

function split_collection {
  col=$1
  inpf=$2
  outf=$3
  probs=$4
  subsets=$5
  if [ ! -f "$RAW_DIR/$inpf" ] ; then
    echo "Cannot find input raw file $RAW_DIR/$inpf for collection $col"
    exit 1
  fi

  CMD_NAME="splitting collection $col"
  LOG_FILE="log_split.$col"
  CMD="scripts/data/split_input.sh -i \"$RAW_DIR/$inpf\"  -o \"$DATA_DIR/input/$col/$outf\" -p \"$probs\" -n \"$subsets\" "
  run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
}

# 2. Splitting & annotating collections
# 2.1 Let's first split collections randomly
split_collection "manner" "manner.xml.bz2" "manner-v2.0" "0.60,0.05,0.15,0.20" "train,dev1,dev2,test"
split_collection "compr" "FullOct2007.xml.bz2" "comprehensive-Oct2007" "0.048,0.0024,0.0094,0.0117,0.9285" "train,dev1,dev2,test,tran"
split_collection "stackoverflow" "PostsNoCode2016-04-28.xml.bz2"  "StackOverflow" "0.048,0.0024,0.0094,0.0117,0.9285" "train,dev1,dev2,test,tran"

# Now let's compute the difference between two collections
CMD_NAME="Computing the diff. between Comprehnsive and Manner"
CMD="scripts/data/diff_collect.sh -i1 \"$RAW_DIR/FullOct2007.xml.bz2\" -i2 \"$RAW_DIR/manner.xml.bz2\" -o input/ComprMinusManner.gz"
LOG_FILE="log_diff"
run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"

echo "Splitting collections into subsets is completed!"

# 2.2 We want smaller subsets, so we replace CPE descriptors with descriptors that limit the number of documents processed

echo "cp $KNN4QA_DIR/scripts/bootstrap/collection_processing_engines.quick/*.xml "

echo "Replaced existing CPE descriptors with descriptors for collection samples"
echo "IMPORTANT NOTE!!!! If something goes awfully wrong you may need to manually 'git checkout' in $CPE_DIR"

# 2.3 An actual annotation process

for col in manner compr ComprMinusManner stackoverflow ; do
  CMD_NAME="annotation pipeline for $col"
  LOG_FILE="log_annot.$col"
  CMD="scripts/uima/run_annot_pipeline.sh $col"
  run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
done

echo "Annotation process is completed!"

# 3 Creating indices

for col in manner compr stackoverflow ; do
  CMD_NAME="creating Lucene index for $col"
  LOG_FILE="log_lucene_index.$col"
  CMD="scripts/index/create_lucene_index.sh $col"
  run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"

  CMD_NAME="creating forward index for $col"
  LOG_FILE="log_inmemfwd_index.$col"
  CMD="scripts/index/create_inmemfwd_index.sh $col"
  run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
done

echo "Lucene and forward indices are created!"

# 4 Creating IBM Model 1 translation files
GIZA_ITER_QTY=5
for col in ComprMinusManner compr stackoverflow ; do
  for field in text text_unlemm ; do
    CMD_NAME="Building IBM Model 1 for $col field $field"
    LOG_FILE="log_model1.${col}_${field}"
    CMD="scripts/giza/create_tran.sh output tran $col tran $field \"$INSTALL_DIR/giza-pp\" 0 $GIZA_ITER_QTY"
    run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
  done
done

echo "IBM Model 1 translation files are created!"

# 5 Building derivative data
for col in ComprMinusManner compr stackoverflow ; do
  CMD_NAME="Generative derivative data for $col"
  LOG_FILE="log_deriv.$col"
  CMD="scripts/data/gen_derivative_data.sh $col"
  run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
done

echo "Derivative data is generated!"

# 6 Let's generate pivots and queries
for col in compr stackoverflow ; do
  CMD_NAME="generative pivots for $col"
  LOG_FILE="log_pivots.$col"
  CMD="scripts/nmslib/gen_nmslib_pivots.sh \"$DATA_DIR\" $col"
  run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
done
echo "Pivots are generated!"

for col in compr stackoverflow ; do
  CMD_NAME="generative queries for $col"
  LOG_FILE="log_queries.$col"
  CMD="scripts/nmslib/gen_nmslib_queries.sh \"$DATA_DIR\" $col"
  run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
done
echo "Queries are generated!"

# 7 Feature experiments 
NUM_PARALLEL_PROC=2
for col in ComprMinusManner compr stackoverflow ; do
  CMD_NAME="running feature experiments for $col (using 2 parallel processes)"
  LOG_FILE="log_feature_exper.$col"
  CMD="scripts/exper/run_feature_experiments.sh manner scripts/exper/feature_desc/test4combinations.txt $NUM_PARALLEL_PROC"
  run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
done
echo "Test feature experiments are finished!"

# 8 Main experiments

for col in compr stackoverflow ; do
  # all brute-force
  TEST_SUBSET="test"
  for script in "scripts/exper/test_nmslib_server_bruteforce_bm25_text.sh" \
                "scripts/exper/test_nmslib_server_bruteforce_cosine_text.sh" \
                "scripts/exper/test_nmslib_server_bruteforce_exper1.sh" \
                "scripts/exper/test_nmslib_server_bruteforce_swgraph.sh" ; do
    CMD_NAME="bruteforce testing with the script $script"
    CMD="$script $col $TEST_SUBSET"
    LOG_FILE="log_${script}.$col"
    run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
  done 

  # indexing scripts with one parameter
  for script in "scripts/exper/test_nmslib_server_napp_bm25_text_${col}.sh" \
                "scripts/exper/test_nmslib_server_napp_exper1_${col}.sh" ; do
    CMD_NAME="indexed testing with the script $script"
    CMD="$script $TEST_SUBSET"
    LOG_FILE="log_${script}.$col"
    run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
  done

  # indexing scripts with two parameters
  for script in "scripts/exper/test_nmslib_server_swgraph.sh" ; do
    CMD_NAME="indexed testing with the script $script"
    CMD="$script $col $TEST_SUBSET"
    LOG_FILE="log_${script}.$col"
    run_cmd_complex "$CMD" "$CMD_NAME" "$LOG_FILE"
  done
done


echo "Main experiments are finished!"
echo "All major work is done!"
restore_desc_exit 0

