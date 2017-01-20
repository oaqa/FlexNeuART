#!/bin/bash
. scripts/common.sh

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
  for field in text text_unlemm bigram srl srl_lab dep wnss ; do
    scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test -solr_file SolrAnswerFile.txt -field $field
    check "scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test -solr_file SolrAnswerFile.txt -field $field"
  done
elif [ "$collect" = "compr" -o  "$collect" = "stackoverflow" ] ; then
  for field in text text_unlemm bigram ; do
    scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test,tran -solr_file SolrAnswerFile.txt -field $field
    check "scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test,tran -solr_file SolrAnswerFile.txt -field $field"
  done
elif [ "$collect" = "squad" ] ; then
  JOINT_NAME=SolrAnswQuestFile.txt
  for d in tran train dev1 dev2 test wiki ; do
    cd $IN_DIR/$d
    check "cd $OUT_DIR/$d"
    if [ "$d" != "wiki" ] ; then
      cat SolrAnswerFile.txt SolrQuestionFile.txt > $JOINT_NAME
      check "cat SolrAnswerFile.txt SolrQuestionFile.txt > $JOINT_NAME"
    else
      rm -f $JOINT_NAME
      ln -s SolrAnswerFile.txt $JOINT_NAME
      check "ln -s SolrAnswerFile.txt $JOINT_NAME"
    fi
    cd -  
    check "cd - "
  done
  for field in qfeat_only text text_qfeat ephyra_spacy ephyra_dbpedia ephyra_allent lexical_spacy lexical_dbpedia lexical_allent ; do
    scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test,tran,wiki -solr_file $JOINT_NAME -field $field
    check "scripts/index/run_inmemfwd_index.sh -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test,tran -solr_file $JOINT_NAME -field $field"
  done
else
  echo "Wrong collection name '$collect'"
  exit 1
fi

