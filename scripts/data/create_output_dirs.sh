#!/bin/bash

# A stupid script to create collection directories
for part in dev1 dev2 test train ; do
  mkdir -p output/manner/$part
done

mkdir -p output/ComprMinusManner/tran

for col in compr stackoverflow wiki_squad squad ; do
  for part in dev1 dev2 test train tran ; do
    mkdir -p output/$col/$part
  done
  if [ "$col" == "wiki_squad" ] ; then
    mkdir -p output/$col/wiki
  fi
done

