#!/bin/bash
../word2vec/trunk/word2vec -train output/stackoverflow/train_text -threads 8 -output word2vec_tran.300 -size 300
