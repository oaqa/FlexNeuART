#!/bin/bash

# This script runs annotation pipelines for a given collection
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow, compr2M, compr_notran, compr2M_notran"
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
  scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test -solr_file SolrAnswerFile.txt
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
elif [ "$collect" = "gov2" ] ; then
  scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs all -solr_file SolrAnswerFile.txt.gz
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
elif [ "$collect" = "compr" ] ; then
  scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test,tran  -solr_file SolrAnswerFile.txt # -exclude_fields "srl,srl_lab,dep,wnss"
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
elif [ "$collect" = "stackoverflow" ] ; then
  scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test,tran  -solr_file SolrAnswerFile.txt -exclude_fields "srl,srl_lab,dep,wnss"
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
elif [ "$collect" = "compr_notran" ] ; then
  # compr_notran is used merely for filtering of embeddings and IBM Model1 translation tables
  # therefore, we create index only for two text fields, excluding everything else 
  IN_DIR="output/compr/"
  scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test  -solr_file SolrAnswerFile.txt  -exclude_fields "bigram,srl,srl_lab,dep,wnss"
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
elif [ "$collect" = "compr2M" ] ; then
  # For compr2M the order of parts MUST be the same as in create_lucene_index.sh !!!
  IN_DIR="output/compr/"
  scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR  -sub_dirs train,dev1,dev2,test,tran -solr_file SolrAnswerFile.txt -n 2000000
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
else
  echo "Wrong collection name '$collect'"
  exit 1
fi

