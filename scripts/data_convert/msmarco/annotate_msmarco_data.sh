#!/bin/bash -e

source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "ANSWER_FILE"
checkVarNonEmpty "QUESTION_FILE"
checkVarNonEmpty "SAMPLE_COLLECT_ARG"

src=$1
if [ "$src" = "" ] ; then
    echo "Specify the source directory (1st arg)"
    exit 1
fi

collect=$2
if [ "$collect" = "" ] ; then
    echo "$SAMPLE_COLLECT_ARG (2d arg)"
    exit 1
fi

python -u scripts/data_convert/msmarco/concatenate_datasets.py \
        --input_datapath "${src}/${INPUT_DATA_SUBDIR}" \
        --output "${src}/${INPUT_DATA_SUBDIR}/concatenated_data.jsonl" 

for DATASPLIT_FOLDER in ${collect}/${INPUT_DATA_SUBDIR}/*; do
    QUESTION_FILE_PATH="${DATASPLIT_FOLDER}/${QUESTION_FILE}"
    if [[ "${DATASPLIT_FOLDER}" != *"bitext"* ]] && test -f "${QUESTION_FILE_PATH}";
    then
        echo "Annotating ${QUESTION_FILE_PATH}"
        python -u scripts/data_convert/msmarco/question_tagging.py \
                "${QUESTION_FILE_PATH}" "${src}/${INPUT_DATA_SUBDIR}/concatenated_data.jsonl" \
                "${DATASPLIT_FOLDER}/tagged_question_ids.json"
    fi
done
