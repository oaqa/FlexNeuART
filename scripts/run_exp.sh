#!/bin/bash

declare -a arr=("random_word_deletion" "random_word_insert" "random_word_swap" "random_character_replace" "keyboard_character_replace")
for aug_method in "${arr[@]}"
do 
    bash train_nn/train_model.sh msmarco_pass cedr_train_pass_50K_200_0_5_0_s1_bitext/text_raw vanilla_bert \
       -batches_per_train_epoch 2 \
       -max_query_val 1 \
       -valid_type last_epoch \
       -epoch_qty 5 \
       -amp \
       -data_augment $aug_method > $aug_method.txt
done
