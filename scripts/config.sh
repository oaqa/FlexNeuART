# TODO change this to the location you want to use as
# the root directory for all collections/indices.
# Each collection is supposed to be stored in respective sub-directories:
# $COLLECT_ROOT
#              <toy-collection>
#                               $INPUT_DATA_SUBDIR
#                                                   train
#                                                   dev
#                                                   test
#                                                   $BITEXT_SUBDIR
#                               $FWD_INDEX_SUBDIR
#                               $LUCENE_INDEX_SUBDIR

COLLECT_ROOT="collections"

FWD_INDEX_SUBDIR="forward_index"
LUCENE_INDEX_SUBDIR="lucene_index"
LUCENE_CACHE_SUBDIR="lucene_cache"

EMBED_SUBDIR="embeddings"
INPUT_DATA_SUBDIR="input_data"
DERIVED_DATA_SUBDIR="derived_data"
FEAT_EXPER_SUBDIR="results/feat_exper"
FINAL_EXPER_SUBDIR="results/final_exper"

BITEXT_SUBDIR="bitext"
GIZA_SUBDIR="giza"
GIZA_ITER_QTY=5

METRIC_TYPE="NDCG@20"
NUM_RAND_RESTART=10

TRAIN_SUBDIR="train"
BITEXT_TRAIN_SUBDIR="train_bitext"


QREL_FILE="qrels.txt"
FAKE_RUN_ID="fake_run"
