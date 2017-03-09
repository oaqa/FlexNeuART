#!/bin/bash
. scripts/common.sh

# This script runs annotation pipelines for a given collection
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow, squad"
  exit 1
fi

store_word_id_seq=$2
store_word_id_seq_param=""
if [ "$store_word_id_seq" = "1" ] ; then
  store_word_id_seq_param=" -store_word_id_seq "
fi

OUT_DIR="memfwdindex/$collect/"
IN_DIR="output/$collect/"

echo "IN_DIR:   $IN_DIR"
echo "OUT_DIR:  $OUT_DIR"
echo "Storing word id seq param: $store_word_id_seq_param"

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
  #for field in text text_unlemm bigram srl srl_lab dep wnss ; do
  for field in text ; do
    scripts/index/run_inmemfwd_index.sh $store_word_id_seq_param -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test -solr_file SolrAnswerFile.txt -field $field
    check "scripts/index/run_inmemfwd_index.sh  $store_word_id_seq_param -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test -solr_file SolrAnswerFile.txt -field $field"
  done
elif [ "$collect" = "compr" -o  "$collect" = "stackoverflow" ] ; then
  for field in text text_unlemm bigram ; do
    scripts/index/run_inmemfwd_index.sh $store_word_id_seq_param -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test,tran -solr_file SolrAnswerFile.txt -field $field
    check "scripts/index/run_inmemfwd_index.sh $store_word_id_seq_param -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test,tran -solr_file SolrAnswerFile.txt -field $field"
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
    scripts/index/run_inmemfwd_index.sh $store_word_id_seq_param -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test,tran,wiki -solr_file $JOINT_NAME -field $field
    check "scripts/index/run_inmemfwd_index.sh $store_word_id_seq_param -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test,tran -solr_file $JOINT_NAME -field $field"
  done
elif [ "$collect" = "gov2" ] ; then
  scripts/index/run_inmemfwd_index.sh -field text -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs all -solr_file SolrAnswerFile.txt.gz
  if [ "$?" != "0" ] ; then
    echo "FAILURE!!!"
    exit 1
  fi
else
  echo "Wrong collection name '$collect'"
  exit 1
fi

