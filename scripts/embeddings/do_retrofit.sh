#!/bin/bash

collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr"
  exit 1
fi

EMBED_LIST=$2
if [ "$EMBED_LIST" = "" ] ; then
  echo "Specify the list of word embedding files (3d arg)"
  exit 1
fi

minProb=$3
if [ "$minProb" = "" ] ; then
  echo "Specify the probability threshold (3d arg)"
  exit 1
fi

THRESHOLD=0.01
MAX_ITER=10

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

lexicon_weighted="WordEmbeddings/$collect/retro_minProb=${minProb}_lex_weighted"
scripts/embeddings/run_create_retrofit_lex.sh -giza_root_dir tran/$collect/ -giza_iter_qty 5 -l $lexicon_weighted  -min_prob $minProb -memindex_dir memfwdindex/$collect/ -f weighted
check "scripts/embeddings/run_create_retrofit_lex.sh -giza_root_dir tran/$collect/ -giza_iter_qty 5 -l $lexicon_weighted  -min_prob $minProb -memindex_dir memfwdindex/$collect/ -f weighted"

lexicon_unweighted="WordEmbeddings/$collect/retro_minProb=${minProb}_lex_unweighted"
scripts/embeddings/run_create_retrofit_lex.sh -giza_root_dir tran/$collect/ -giza_iter_qty 5 -l $lexicon_unweighted  -min_prob $minProb -memindex_dir memfwdindex/$collect/ -f unweighted
check "scripts/embeddings/run_create_retrofit_lex.sh -giza_root_dir tran/$collect/ -giza_iter_qty 5 -l $lexicon_unweighted  -min_prob $minProb -memindex_dir memfwdindex/$collect/ -f unweighted"

for efile in $EMBED_LIST ; do 
  # Weighted retrofitting
  outFile="WordEmbeddings/$collect/${efile}_retro_weighted_minProb=${minProb}.txt"
  scripts/embeddings/retrofit_weighted.py  -i WordEmbeddings/${collect}/${efile}.txt -l $lexicon_weighted -o $outFile --eps $THRESHOLD -n $MAX_ITER
  check "scripts/embeddings/retrofit_weighted.py  -i WordEmbeddings/${collect}/${efile}.txt -l $lexicon_weighted -o $outFile --eps $THRESHOLD -n $MAX_ITER"
  # Unweighted retrofitting
  outFile="WordEmbeddings/$collect/${efile}_retro_unweighted_minProb=${minProb}.txt"
  scripts/embeddings/retrofit_weighted.py  -i WordEmbeddings/${collect}/${efile}.txt -l $lexicon_unweighted -o $outFile --eps $THRESHOLD -n $MAX_ITER
  check "scripts/embeddings/retrofit_weighted.py  -i WordEmbeddings/${collect}/${efile}.txt -l $lexicon_unweighted -o $outFile --eps $THRESHOLD -n $MAX_ITER"
done
