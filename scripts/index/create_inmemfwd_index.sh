#!/bin/bash -e
. scripts/common.sh

# This script runs annotation pipelines for a given collection
collect=$1
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (1st arg): manner, compr, stackoverflow, wiki_squad, squad"
  exit 1
fi

store_word_id_seq=$2
store_word_id_seq_param=""
if [ "$store_word_id_seq" = "" ] ; then
  echo "Specify the flag to store word sequence (2d arg): 1 or 0"
  exit 1
fi
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
#echo "Removing previous index (if exists)"
#rm -rf "$OUT_DIR"/*
echo "=========================================================================="

if [ "$collect" = "manner" ] ; then
  for field in text text_unlemm ; do
    scripts/index/run_inmemfwd_index.sh $store_word_id_seq_param -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test -solr_file SolrAnswerFile.txt -field $field
  done
elif [ "$collect" = "compr" -o  "$collect" = "stackoverflow" ] ; then
  for field in text text_unlemm ; do
    scripts/index/run_inmemfwd_index.sh $store_word_id_seq_param -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs train,dev1,dev2,test,tran -solr_file SolrAnswerFile.txt -field $field
  done
elif [ "$collect" = "clueweb09" ] ; then
  #for field in title linkText text ; do
  for field in linkText ; do
    scripts/index/run_inmemfwd_index.sh $store_word_id_seq_param -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs all -solr_file ClueWeb09Proc.xml.gz -field $field
  done
elif [ "$collect" = "squad" -o "$collect" = "wiki_squad" ] ; then
  JOINT_NAME=SolrAnswQuestFile.txt
  if [ "$collect" = "wiki_squad" ] ; then
    wiki_part="wiki"
    part_list="train,dev1,dev2,test,tran,wiki"
  else
    part_list="train,dev1,dev2,test,tran"
  fi
  for d in tran train dev1 dev2 test $wiki_part ; do
    cd $IN_DIR/$d
    if [ "$d" != "wiki" ] ; then
      cat SolrAnswerFile.txt SolrQuestionFile.txt > $JOINT_NAME
    else
      rm -f $JOINT_NAME
      ln -s SolrAnswerFile.txt $JOINT_NAME
    fi
    cd -  
  done
  for field in text text_unlemm ; do
    if [ "$field" == "text" ] ; then
      SOURCE_NAME="SolrAnswerFile.txt"
    else
      SOURCE_NAME="$JOINT_NAME"
    fi
    scripts/index/run_inmemfwd_index.sh $store_word_id_seq_param -root_dir $IN_DIR  -index_dir $OUT_DIR -sub_dirs $part_list -solr_file $SOURCE_NAME -field $field
  done
else
  echo "Wrong collection name '$collect'"
  exit 1
fi

