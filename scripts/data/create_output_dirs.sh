#!/bin/bash

# A stupid script to create collection directories
for part in dev1 dev2 test train ; do
  mkdir -p output/manner/$part
done

mkdir -p output/ComprMinusManner/tran

for part in dev1 dev2 test train1 train2 tran ; do
  mkdir -p output/compr/$part
done


for part in dev1 dev2 test train tran ; do
  mkdir -p output/stackoverflow/$part
done
