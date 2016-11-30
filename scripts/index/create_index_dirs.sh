#!/bin/bash

# A stupid script to create index directories
for d in compr manner stackoverflow ; do
  mkdir -p memfwdindex/$d
  mkdir -p lucene_index/$d
  mkdir -p galago_index/$d
done
