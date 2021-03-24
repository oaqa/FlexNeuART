#!/bin/bash -e

source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "INPUT_RAW_SUBDIR"
checkVarNonEmpty "ANSWER_FILE"
checkVarNonEmpty "QUESTION_FILE"
checkVarNonEmpty "SAMPLE_COLLECT_ARG"

collect=$1
if [ "$collect" = "" ] ; then
  echo "$SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

dataDownloadDir="${COLLECT_ROOT}/${collect}/${INPUT_RAW_SUBDIR}"

python -u scripts/data_convert/msmarco/concatenate_datasets.py \
        --input_datapath "${dataDownloadDir}/" \
        --output "${dataDownloadDir}/concatenated_data.jsonl"

for inpSubDir in ${collect}/${INPUT_DATA_SUBDIR}/*; do
    questFilePath="${inpSubDir}/${QUESTION_FILE}"
    # Produce annotations for all available files
    if [ -f "${questFilePath}" ] ;
    then
        echo "Annotating ${questFilePath}"
        python -u scripts/data_convert/msmarco/question_tagging.py \
                "${questFilePath}" \
                "${dataDownloadDir}/concatenated_data.jsonl" \
                "${inpSubDir}/tagged_question_ids.json"
    fi
done
