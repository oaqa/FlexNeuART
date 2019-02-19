#!/bin/bash -e
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow, squad, clueweb09"
  exit 1
fi

MAX_NUM_DOC=50
MAX_NUM_QUERY=100
EPS_DIFF=1e-5

cdir=scripts/qa
scripts/data/run_check_dense_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text_unprunned.json -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -memindex_dir memfwdindex/$collect/ -q output/$collect/dev1/SolrQuestionFile.txt -giza_root_dir tran/$collect/ -eps_diff $EPS_DIFF -embed_dir WordEmbeddings/$collect

#scripts/data/run_check_dense_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text+embed=text_unprunned.json -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -memindex_dir memfwdindex/$collect/ -q output/$collect/dev1/SolrQuestionFile.txt -giza_root_dir tran/$collect/ -eps_diff $EPS_DIFF -embed_dir WordEmbeddings/$collect
