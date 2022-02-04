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

"""
    A configurable pipeline module that can be customized to parse one or more
    IR-datasets fields into a multi-field JSONL.

     A configurable pipeline to process a generic IR-dataset: https://ir-datasets.com/

     Simplified Data Wrangling with ir_datasets.
     MacAvaney, S., Yates, A., Feldman, S., Downey, D., Cohan, A., & Goharian, N. (2021)

     In proceedings of SIGIR 2021
"""
import gc
from copy import copy

import ir_datasets

from flexneuart.config import DOCID_FIELD
from flexneuart.ir_datasets import proc_pipeline_registry
from flexneuart.io.json import get_val_err_msg_miss

#
#  A configuration JSON contains multiple entries each of which has the following attributes
#  (maybe we can use JSON SCHEMA at some point):
#
#  part_name (in the input folder)
#  dataset_name (IR dataset / subset name)
#  is_query
#  src_attributes - the name of source attributes. If there's HTML stored in the 'body' field,
    #                      make to include 'body_content_type' and 'http_headers'
#  ir_datasets [
#
#     Pipeline stages is an array of arrays.
#       i. Top-level arrays corresponds to processing stages.
#       ii. Each stage can have one or more component that all process output from the previous stage.

#     pipeline_stages : [
#       [ Component1 : { argument array }, Component 2 : { argument array } , ... ],
#       [ Component3 : { argument array }, ],
#                ...
#    ]
# ]
#
#

DEBUG=False

if DEBUG:
    import json

class PipelineAttrGenerator:
    def __init__(self, pipeline, data_iter_obj):
        self.pipeline = pipeline
        self.iter_obj = data_iter_obj

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        return self.pipeline.extract_src_attributes(next(self.iter_obj))


class Pipeline:

    def __init__(self, dataset_name : str, part_name : str,
                 is_query : bool, src_attributes : list, stage_arr : list):
        """Constructor

            :param dataset_name:    the name of the dataset or its subset (should match ir-datasets)
            :param part_name:       the name of the input part (it defines the sub-directory where we output data)
            :param is_query:        true for queries
            :param src_attributes:  an array of attributes to extract from an ir-datasets object
            :param stage_arr:       an array of arrays of processing components grouped together by stages

        """
        self.dataset_name = dataset_name
        self.is_query = is_query
        add_attributes = ['query_id'] if is_query else ['doc_id']
        self.src_attributes = src_attributes + add_attributes
        self.stage_arr = stage_arr
        self.part_name = part_name
        self.comp_cache = {}

    def dataset_iterator(self):
        dataset = ir_datasets.load(self.dataset_name)
        if self.is_query:
            iter_obj = dataset.queries_iter()
        else:
            iter_obj = dataset.docs_iter()

        return PipelineAttrGenerator(self, iter_obj)
            

    def finish_processing(self):
        """This function must be called to free memory."""
        self.comp_cache = {}
        gc.collect()

    def __call__(self, input_dict):
        """Process an input object.

        :param input_dict: a dictionary of input attributes generted by the funcioion dataset_iterator
        :return: a dictionary of data fields including a mandatory field DOCID_FIELD
        """

        if DEBUG:
            print(f'INPUT')
            print(json.dumps(input_dict, indent=2))

        curr_dict = copy(input_dict)

        # Looping over processing stages
        for stage_idx, stage in enumerate(self.stage_arr):
            # each stage starts from the clear output
            # if the object is not explicitly passed/processed by the stage components, it's discarded

            output_dict = {}

            # Processing component loop
            for comp_idx, (comp_class, comp_args) in enumerate(stage):
                comp_key = f'{stage_idx}_{comp_idx}'
                if not comp_key  in self.comp_cache:
                    self.comp_cache[comp_key] = comp_class(**comp_args)
                comp = self.comp_cache[comp_key]

                # Pass input through each processing component
                for k, v in comp(curr_dict).items():
                    if k in output_dict:
                        raise Exception(f'Repeating field {k} stage {stage_idx + 1} component {comp_idx+1}')
                    output_dict[k] = v

            curr_dict = output_dict

            if DEBUG:
                print(f'Stage #{stage_idx+1}')
                print(json.dumps(curr_dict, indent=2))

        if self.is_query:
            output_dict[DOCID_FIELD] = input_dict['query_id']
        else:
            output_dict[DOCID_FIELD] = input_dict['doc_id']

        return output_dict


    @staticmethod
    def parse_config(parsed_json_config):
        """"
            :param parsed_json_config: a parsed JSON.

            :return an array of Pipeline objects
        """
        res = []

        seen = set()
        assert type(parsed_json_config) == list, "Top-level configuration should contain an array of definitions!"
        for part_idx, part_conf in enumerate(parsed_json_config):
            part_name = None
            try:
                part_name = get_val_err_msg_miss(part_conf, 'part_name', [str])
                if part_name in seen:
                    raise Exception('Repeating input part:' + part_name)

                dataset_name = get_val_err_msg_miss(part_conf, 'dataset_name', [str])
                is_query = bool(get_val_err_msg_miss(part_conf, 'is_query', [int, bool]))
                src_attributes = get_val_err_msg_miss(part_conf, 'src_attributes', [list])
                pipeline = get_val_err_msg_miss(part_conf, 'pipeline', [list])
                assert type(pipeline) == list, f"Pipeline definition for part {part_name} is not an array!"

                stage_arr = []
                for stage_idx, stage_def in enumerate(pipeline):
                    assert type(stage_def) == list, f"Stage # {stage_idx+1} part {part_name} ir_datasets is not an array!"
                    comp_arr = []
                    for comp_def in stage_def:
                        assert type(comp_def) == dict
                        comp_name = get_val_err_msg_miss(comp_def, 'name', [str])
                        comp_arg = get_val_err_msg_miss(comp_def, 'args', [dict], attr_default={})
                        if not comp_name in proc_pipeline_registry.registered:
                            raise Exception(f'we have no component with the name {comp_name}')
                        comp_class = proc_pipeline_registry.registered[comp_name]
                        # Don't instantiate a component, until it's actually used
                        comp_arr.append((comp_class, comp_arg))
                    stage_arr.append(comp_arr)

            except Exception as e:
                if part_name is not None:
                    raise Exception(f'Exception parsing part # {part_idx+1} part name: {part_name}, ' + (str(e)))
                else:
                    raise Exception(f'Exception parsing part # {part_idx+1}, ' + str(e))

            res.append(Pipeline(dataset_name=dataset_name,
                                part_name=part_name,
                                is_query=is_query,
                                src_attributes=src_attributes,
                                stage_arr=stage_arr))

        return res

    def extract_src_attributes(self, input_obj):
        """Extract attributes from an input object and return them as a key-val dictionary"""
        res = {}
        for field_name in self.src_attributes:
            if not hasattr(input_obj, field_name):
                raise Exception(f'Missing input attribute: {field_name} dataset: {self.dataset_name}')
            res[field_name] = getattr(input_obj, field_name)

        return res
