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
import multiprocessing
import sys
import glob
import os

"""
    This file contains a number of miscellaneous. helper functions.
"""

def configure_classpath(source_root):
    """Add the latest FlexNeuART jar to the path.

       This function is based on Pyserini code https://github.com/castorini/pyserini

    :param source_root: source root
    """

    from jnius_config import set_classpath

    paths = glob.glob(os.path.join(source_root, 'FlexNeuART-*-fatjar.jar'))
    if not paths:
        raise Exception('No matching jar file found in {}'.format(os.path.abspath(source_root)))

    latest = max(paths, key=os.path.getctime)
    set_classpath(latest)

#
# The Register class is taken from OpenNIR:
# https://github.com/Georgetown-IR-Lab/OpenNIR
#
# (c) Georgetown IR lab & Carnegie Mellon University
#
# It's distributed under the MIT License
# MIT License is compatible with Apache 2 license for the code in this repo.
#


"""A decorator that used to register classes."""
class Registry:
    def __init__(self, default: str = None):
        self.registered = {}
        self.default = default

    def register(self, name):
        registry = self

        def wrapped(fn):
            registry.registered[name] = fn
            fn.name = name
            return fn
        return wrapped


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


def sync_out_streams():
    """Just flush all stdin and stderr to make streams go in sync"""
    sys.stderr.flush()
    sys.stdout.flush()

