bash train_nn/train_model.sh msmarco_pass cedr_train_pass_50K_200_0_5_0_s1_bitext/text_raw vanilla_bert \
    -batches_per_train_epoch 16 \
    -max_query_val 100 \
    -amp \
    -data_augment random_word_deletion
