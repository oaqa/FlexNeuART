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
import torch
import random
import numpy
import tempfile
import os
import multiprocessing
import json
import sys

PYTORCH_DISTR_BACKEND='gloo'

def set_all_seeds(seed):
    """Just set the seed value for common packages including the standard random."""
    print(f'Setting the seed to {seed}')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def enable_spawn():
    """Enable light-weight children. Plus, it is
       a must-use process createion mode for multi-GPU training.
    """
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass


def join_and_check_stat(proc):
    """Join the process and check its status:
       Raise an exception when a sub-process exits abnormally (exit status != 0).
    """
    proc.join()
    if proc.exitcode != 0:
        raise Exception('A process exited abnormally with code:' + str(proc.exitcode))


def read_json(file_name):
    """Read and parse JSON file

    :param file_name: JSON file name
    """
    with open(file_name) as f:
        data = f.read()

    return json.loads(data)


def save_json(file_name, data, indent=4):
    """Save JSON data

    :param file_name:   output file name
    :param data:        JSON data
    :param indent:      JSON indentation
    """
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=indent)


def sync_out_streams():
    """Just flush all stdin and stderr to make streams go in sync"""
    sys.stderr.flush()
    sys.stdout.flush()


def create_temp_file():
    """"Create a temporary file
    :return temporary file name
    """
    f, file_name = tempfile.mkstemp()
    os.close(f)
    return file_name
