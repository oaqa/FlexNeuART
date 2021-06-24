#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import sys
import re
import json

sys.path.append('.')

from scripts.data_convert.msmarco.similarity_funcs import is_equal, tokenized_equal


def annotate_questions_using_msmarco_dataset(passage_dataset, qa_dataset, similarity_func):
    """Uses the MSMarco QA dataset to annontate the passage ranking
       dataset with query type tags based on a similarity metric.

       returns a dictionary with query_type as key and a list of 
       corresponding DOCNOs as values
    """
    question_type_2_docno = {"LOCATION": [],
                             "DESCRIPTION": [],
                             "ENTITY": [],
                             "PERSON": [],
                             "NUMERIC": []}
    
    for question in passage_dataset:
        question_type = find_question_type_from_dataset(question, qa_dataset, similarity_func)
        if question_type:
            question_type_2_docno[question_type].append(question["DOCNO"])
    return question_type_2_docno


def find_question_type_from_dataset(question, qa_dataset, similarity_func):
    """Returns query_type of the datapoint from ms_marco QA dataset
       for the first match.
    """
    for datapoint in qa_dataset:
        if similarity_func(question["text_raw"], datapoint["query"]):
            return datapoint["query_type"]


def load_jsonl(filepath):
    datapoints = []
    with open(filepath) as f:
        for line in f:
            datapoints.append(json.loads(line))
    return datapoints


if __name__ == "__main__":
    
    passage_dataset_filepath = sys.argv[1]
    qa_dataset_filepath = sys.argv[2]
    outfilename = sys.argv[3]

    passage_dataset = load_jsonl(passage_dataset_filepath)
    qa_dataset = load_jsonl(qa_dataset_filepath)

    question_type_annotation = annotate_questions_using_msmarco_dataset(passage_dataset, qa_dataset, is_equal)

    with open(outfilename, "w") as f:
        json.dump(question_type_annotation, f)
