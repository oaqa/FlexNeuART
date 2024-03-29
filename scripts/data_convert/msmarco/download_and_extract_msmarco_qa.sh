#!/bin/bash -e

# Downloads MS MARCO QA and NLG dataset v2.1
# to collections directory

source ./common_proc.sh
source ./config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_RAW_SUBDIR"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

dataDownloadDir="${COLLECT_ROOT}/${collect}/${INPUT_RAW_SUBDIR}"

if [ ! -d "${dataDownloadDir}" ] ; then
  mkdir -p "${dataDownloadDir}"
fi

cd "${dataDownloadDir}"

if test -f "SUCCESS";
then
  echo "Data downloaded"
  exit 0
fi

for split in train_v2.1.json.gz \
             dev_v2.1.json.gz \
             eval_v2.1_public.json.gz \
             ; do
  uri="https://msmarco.blob.core.windows.net/msmarco/${split}"
  echo "Download ${uri}"
  wget "${uri}"
done

for file in ./*.gz;
do
  gunzip "${file}"
done

touch "SUCCESS"
