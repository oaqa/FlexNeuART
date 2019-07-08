#!/bin/bash -e
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg), e.g.,  squad"
  exit 1
fi

MAX_NUM_DOC=50
MAX_NUM_QUERY=100
EPS_DIFF=1e-5

cdir=scripts/qa

scripts/data/run_check_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text_unpruned.json -model_file $cdir/bm25=text+model1=text.model -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -memindex_dir memfwdindex/$collect/ -q output/$collect/dev1/SolrQuestionFile.txt -giza_root_dir tran/$collect/ -eps_diff $EPS_DIFF -embed_dir WordEmbeddings/$collect

scripts/data/run_check_dense_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text+embed=text.json -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -memindex_dir memfwdindex/$collect/ -q output/$collect/dev1/SolrQuestionFile.txt -giza_root_dir tran/$collect/ -eps_diff $EPS_DIFF -embed_dir WordEmbeddings/$collect

scripts/data/run_check_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text.json -model_file $cdir/bm25=text+model1=text.model -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -memindex_dir memfwdindex/$collect/ -q output/$collect/dev1/SolrQuestionFile.txt -giza_root_dir tran/$collect/ -eps_diff $EPS_DIFF -embed_dir WordEmbeddings/$collect

scripts/data/run_check_dense_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text+embed=text.json -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -memindex_dir memfwdindex/$collect/ -q output/$collect/dev1/SolrQuestionFile.txt -giza_root_dir tran/$collect/ -eps_diff $EPS_DIFF -embed_dir WordEmbeddings/$collect

scripts/data/run_check_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text+embed=text.json -model_file $cdir/bm25=text+model1=text+embed=text.model -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -memindex_dir memfwdindex/$collect/ -q output/$collect/dev1/SolrQuestionFile.txt -giza_root_dir tran/$collect/ -eps_diff $EPS_DIFF -embed_dir WordEmbeddings/$collect

scripts/data/run_check_dense_sparse_export_scores.sh -extr_json $cdir/bm25=text+model1=text+embed=text.json -model_file $cdir/bm25=text+model1=text+embed=text.model -max_num_doc $MAX_NUM_DOC -max_num_query $MAX_NUM_QUERY -memindex_dir memfwdindex/$collect/ -q output/$collect/dev1/SolrQuestionFile.txt -giza_root_dir tran/$collect/ -eps_diff $EPS_DIFF -embed_dir WordEmbeddings/$collect

echo "ALL CHECKS ARE DONE!"
