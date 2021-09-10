#!/bin/bash -e

source ./common_proc.sh
source ./config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "INPUT_RAW_SUBDIR"
checkVarNonEmpty "QUESTION_FILE_JSONL"
checkVarNonEmpty "SAMPLE_COLLECT_ARG"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

dataDownloadDir="${COLLECT_ROOT}/${collect}/${INPUT_RAW_SUBDIR}"

python -u ./data_convert/msmarco/concatenate_datasets.py \
        --input_datapath "${dataDownloadDir}/" \
        --output "${dataDownloadDir}/concatenated_data.jsonl"


for inpSubDir in "${COLLECT_ROOT}/${collect}/${INPUT_DATA_SUBDIR}/"/* ; do
    questFilePath="${inpSubDir}/${QUESTION_FILE_JSONL}"
    echo "Checking existence of the file: $questFilePath"
    # Produce annotations for all available files
    if [ -f "${questFilePath}" ] ;
    then
        echo "Annotating ${questFilePath}"
        python -u ./data_convert/msmarco/question_tagging.py \
                "${questFilePath}" \
                "${dataDownloadDir}/concatenated_data.jsonl" \
                "${inpSubDir}/tagged_question_ids.json"
    fi
done
