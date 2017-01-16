#!/bin/bash

# This script runs annotation pipelines for a given collection
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow, squad"
  exit 1
fi

OUT_DIR="memfwdindex/$collect/"
IN_DIR="output/$collect/"

if [ ! -d "$OUT_DIR" ] ; then
  echo "The output directory '$OUT_DIR' doesn't exist!"
  exit 1
fi

echo "=========================================================================="
echo "Output directory: $OUT_DIR"
echo "Removing previous index (if exists)"
rm -f "$OUT_DIR"/*
echo "=========================================================================="

if [ "$collect" = "manner" ] ; then
  scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test -solr_file SolrAnswerFile.txt -exclude_fields qfeat_all
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
elif [ "$collect" = "compr" ] ; then
  scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test,tran  -solr_file SolrAnswerFile.txt -exclude_fields "srl,srl_lab,dep,wnss,qfeat_all"
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
elif [ "$collect" = "stackoverflow" ] ; then
  scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test,tran  -solr_file SolrAnswerFile.txt -exclude_fields "srl,srl_lab,dep,wnss,qfeat_all"
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
elif [ "$collect" = "squad" ] ; then
  scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test,tran,wiki  -solr_file SolrQuestionAnswerFile.txt -exclude_fields "bigram,srl,srl_lab,dep,wnss,text_unlemm"
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
else
  echo "Wrong collection name '$collect'"
  exit 1
fi

