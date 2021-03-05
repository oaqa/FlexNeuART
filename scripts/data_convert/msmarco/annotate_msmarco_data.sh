#!/bin/bash

# Download the msmarco qa dataset in the collections folder
# compile the train, dev and eval data into a single file

COLLECTIONS_PATH="../../../collections"
QA_DATAPATH="${COLLECTIONS_PATH}/msmarco_qa"

rm "${QA_DATAPATH}/transformed_data.jsonl"

./transform_ms_marco_qa_dataset.py "${QA_DATAPATH}/train_v2.1.json" "${QA_DATAPATH}/transformed_data.jsonl"
./transform_ms_marco_qa_dataset.py "${QA_DATAPATH}/dev_v2.1.json" "${QA_DATAPATH}/transformed_data.jsonl"
./transform_ms_marco_qa_dataset.py "${QA_DATAPATH}/eval_v2.1_public.json" "${QA_DATAPATH}/transformed_data.jsonl"

PASSAGE_DATAPATH="${COLLECTIONS_PATH}/msmarco_doc/input_data"

mkdir -p "${COLLECTIONS_PATH}/question_types/"

for datasplit in ${PASSAGE_DATAPATH}/*;
    do
        filename=$(basename "${datasplit}" .json)
        if [ ${filename} != "bitext" ] && [ ${filename} != "docs" ]
            then
                echo "annotating ${filename}"
                ./question_tagging.py "${datasplit}/QuestionFields.jsonl" "${QA_DATAPATH}/transformed_data.jsonl"  "${COLLECTIONS_PATH}/question_types/${filename}_em.json"
        fi
    done