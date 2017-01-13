#!/bin/bash

# A stupid script to create collection directories
for part in dev1 dev2 test train ; do
  mkdir -p output/manner/$part
done

mkdir -p output/ComprMinusManner/tran

for col in compr stackoverflow squad ; do
  for part in dev1 dev2 test train tran ; do
    mkdir -p output/$col/$part
  done
  if [ "$col" == "squad" ] ; then
    mkdir -p output/$col/wiki
  fi
done

