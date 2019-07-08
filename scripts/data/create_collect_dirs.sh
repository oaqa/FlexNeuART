#!/bin/bash
collect=$1

[ "$collect" != "" ] || { echo "Specify collection sub-directory!" ; exit 1 ; } 

# A stupid script to create index & data directories
mkdir -p $collect/forward_index
mkdir -p $collect/lucene_index
mkdir -p $collect/input_data
mkdir -p $collect/word_embeddings
