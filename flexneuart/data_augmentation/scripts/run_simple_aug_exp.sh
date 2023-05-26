#!/bin/bash

exp_name=$1

if [ -z "$exp_name" ]
	then
		echo "Please provide exp name"
		exit
fi

script_home=/home/ubuntu/efs/capstone/data_aug/FlexNeuART/scripts
cd $script_home

output_dir=/home/ubuntu/efs/capstone/data_aug/experiments/$exp_name
mkdir -p $output_dir

export CONF_PATH=model_conf/ernie_large_ce_loss.json
export COLLECT_ROOT=/home/ubuntu/efs/capstone/data_aug/data
export DATA_FOLDER=msmarco
export DATA_FILE=inpars_subsample

#declare -a arr=("random_word_deletion" "random_word_insert" "random_word_swap" "random_character_replace" "keyboard_character_replace" "synonym_word_replacement" "random_character_insertion" "random_character_deletion")
declare -a arr=("random_word_swap" "synonym_word_replacement")
for aug_method in "${arr[@]}"
do 
    bash train_nn/train_model.sh $DATA_FOLDER $DATA_FILE vanilla_bert \
	    -json_conf $CONF_PATH \
	    -max_query_val 1 \
	    -batches_per_train_epoch 1 \
	    -epoch_qty 2 \
	    -amp \
	    -add_exper_subdir $exp_name/$aug_method \
	    -data_augment $aug_method | tee $output_dir/$aug_method.txt
done
