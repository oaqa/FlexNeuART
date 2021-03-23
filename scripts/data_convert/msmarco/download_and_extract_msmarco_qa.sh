#!/bin/bash -e

# Downloads MS MARCO QA and NLG dataset v2.1
# to collections directory

source scripts/config.sh

DATASET_FOLDER_NAME="${1}"

DATA_DOWNLOAD_DIR="${COLLECT_ROOT}/${DATASET_FOLDER_NAME}/${INPUT_DATA_SUBDIR}"

mkdir -p "${DATA_DOWNLOAD_DIR}"

cd "${DATA_DOWNLOAD_DIR}"

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
