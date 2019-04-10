#!/bin/bash -e
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow, wiki_squad, squad, clueweb09"
  exit 1
fi

IN_DIR="output/$collect/"
OUT_DIR="lucene_index/$collect"

if [ ! -d "$OUT_DIR" ] ; then
  echo "Directory (or a link to a directory) $OUT_DIR doesn't exist"
  exit 1
fi
echo "=========================================================================="
echo "Output directory: $OUT_DIR"
echo "Removing previous index (if exists)"
rm -rf "$OUT_DIR"/*
echo "=========================================================================="

if [ "$collect" = "manner" ] ; then
  scripts/index/run_lucene_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test -solr_file SolrAnswerFile.txt
elif [ "$collect" = "compr" ] ; then
  scripts/index/run_lucene_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test,tran  -solr_file SolrAnswerFile.txt
elif [ "$collect" = "stackoverflow" ] ; then
  scripts/index/run_lucene_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test,tran  -solr_file SolrAnswerFile.txt
elif [ "$collect" = "squad" ] ; then
  scripts/index/run_lucene_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test,tran -solr_file SolrAnswerFile.txt
elif [ "$collect" = "wiki_squad" ] ; then
  scripts/index/run_lucene_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test,tran,wiki  -solr_file SolrAnswerFile.txt
elif [ "$collect" = "clueweb09" ] ; then
  scripts/index/run_lucene_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs all -solr_file ClueWeb09Proc.xml.gz 
else
  echo "Wrong collection name '$collect'"
  exit 1
fi
