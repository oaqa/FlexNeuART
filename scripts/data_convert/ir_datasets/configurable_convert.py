#!/usr/bin/env python
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
 A configurable processing of a generic IR-datasets: https://ir-datasets.com/

 Simplified Data Wrangling with ir_datasets.
 MacAvaney, S., Yates, A., Feldman, S., Downey, D., Cohan, A., & Goharian, N. (2021)

 In proceedings of SIGIR 2021

"""
import os
import ir_datasets
import json
import argparse
import multiprocessing

from tqdm import tqdm

from flexneuart import configure_classpath
from flexneuart.models.train.distr_utils import enable_spawn

configure_classpath()

from flexneuart.io import FileWrapper
from flexneuart.config import QUESTION_FILE_JSON, ANSWER_FILE_JSONL_GZ, QREL_FILE

from flexneuart.ir_datasets.pipeline import Pipeline
from flexneuart.io.json import read_json
from flexneuart.io.qrels import qrel_entry2_str, QrelEntry
    
class ParseWorker:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, obj):
        try:
            return json.dumps(self.pipeline(obj)) + '\n'
        except Exception as e:
            print('Exception when processing: ' + str(e))
            return e

def main():


    parser = argparse.ArgumentParser(description='Convert an IR dataset collection.')

    parser.add_argument('--config', metavar='JSON configuration',
                        help='JSON configuration file that describes the processing steps',
                        type=str, required=True)
    parser.add_argument('--output_root', metavar='The root output directory',
                        type=str, required=True)
    # Default is: Number of cores minus one for the spaning process
    parser.add_argument('--proc_qty', metavar='# of processes', help='# of NLP processes to span',
                        type=int, default=multiprocessing.cpu_count() - 1)
    
    
    args = parser.parse_args()
    print(args)
    
    parsed_config = read_json(args.config)
    

    proc_qty = args.proc_qty
    print(f'Spanning {proc_qty} processes')
    pool = multiprocessing.Pool(processes=proc_qty)

    error_qty = 0

    for e in Pipeline.parse_config(parsed_config):
        part_processor : Pipeline = e

        out_dir = os.path.join(args.output_root, part_processor.part_name)
        os.makedirs(out_dir, exist_ok=True)
        if part_processor.is_query:
            out_data_file = QUESTION_FILE_JSON
        else:
            out_data_file = ANSWER_FILE_JSONL_GZ

        # Process document or QUERIES, here it makes sense to juse multiple processes
        with FileWrapper(os.path.join(out_dir, out_data_file), 'w') as f_out_data:
            obj_id = 0
    
            worker = ParseWorker(pipeline=part_processor)

            if proc_qty == 1:
                map_obj = map(worker, part_processor.dataset_iterator())
            else:
                # The size of the buffer is a bit adhoc, but it usually works well for documents with HTML,
                # where processing is the slowest operation
                map_obj = pool.imap(worker, part_processor.dataset_iterator(), proc_qty * 16)

            for res in tqdm(map_obj,
                    f'converting part {part_processor.part_name} query? {part_processor.is_query}, errors: {error_qty}'):
                obj_id = obj_id + 1

                if type(res) == str:
                    f_out_data.write(res)
                else:
                    print(f'Failed to convert object # {obj_id}: ' + str(res))
                    error_qty += 1
                    continue

    
    
            part_processor.finish_processing()
    
            print(f'Processed {obj_id} objects, errors: {error_qty}')

        # Save QRELs. QRELs are very easy to process, not need to span extra processes
        if part_processor.is_query:
            with FileWrapper(os.path.join(out_dir, QREL_FILE), 'w') as f_out_qrel:

                dataset = ir_datasets.load(part_processor.dataset_name)
                # Although QREL type entries can be different in IR datasets (e.g. TrecQrel doesn't inherit from BaseQrel),
                # they all seem to have the following fields:
                #     query_id: str
                #     doc_id: str
                #     relevance: int
                for qrel_orig in tqdm(dataset.qrels_iter(), 'saving QRELs'):
                    f_out_qrel.write(qrel_entry2_str(QrelEntry(query_id=qrel_orig.query_id,
                                                               doc_id=qrel_orig.doc_id,
                                                               rel_grade=qrel_orig.relevance)))
                    f_out_qrel.write('\n')


if __name__ == '__main__':

    enable_spawn()
    main()

