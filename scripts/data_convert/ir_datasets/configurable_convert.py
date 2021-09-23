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
import sys
import json
import argparse
import multiprocessing

from tqdm import tqdm

from flexneuart import configure_classpath

configure_classpath()

from flexneuart.io import FileWrapper
from flexneuart.config import IMAP_PROC_CHUNK_QTY, REPORT_QTY, QUESTION_FILE_JSON, ANSWER_FILE_JSON


from flexneuart.ir_datasets.pipeline import Pipeline
from flexneuart.io.json import read_json


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


class ParseWorker:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, obj):
        try:
            return json.dumps(self.pipeline(obj)) + '\n'
        except Exception as e:
            print('Exception when processing: ' + str(e))
            return None



proc_qty = args.proc_qty
print(f'Spanning {proc_qty} processes')
pool = multiprocessing.Pool(processes=proc_qty)


for e in Pipeline.parse_config(parsed_config):
    part_processor : Pipeline = e

    out_dir = os.path.join(args.output_root, part_processor.part_name)
    os.makedirs(out_dir, exist_ok=True)
    if part_processor.is_query:
        out_file = QUESTION_FILE_JSON
    else:
        out_file = ANSWER_FILE_JSON

    with FileWrapper(os.path.join(out_dir, out_file), 'w') as out_file:
        obj_id = 0

        worker = ParseWorker(pipeline=part_processor)

        for doc_str in tqdm(pool.imap(worker, part_processor.dataset_iterator(), IMAP_PROC_CHUNK_QTY),
                                      f'converting part {part_processor.part_name} query? {part_processor.is_query}'):
            obj_id = obj_id + 1
            if doc_str is not None:
                out_file.write(doc_str)
            else:
                print(f'Failed to convert object # {obj_id}')
                sys.exit(1)


        part_processor.finish_processing()

        print('Processed %d objects' % obj_id)

