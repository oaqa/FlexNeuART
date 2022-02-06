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
    This file contains a number of miscellaneous helper functions.
"""
import torch
import random
import numpy
import sys


def set_all_seeds(seed):
    """Just set the seed value for common packages including the standard random."""
    print(f'Setting the seed to {seed}')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def sync_out_streams():
    """Just flush all stdin and stderr to make streams go in sync"""
    sys.stderr.flush()
    sys.stdout.flush()


def clear_line_console():
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")


class DictToObject(object):
    """Create an object from a dictionary, but not recursively"""
    def __init__(self, input_dict):
        """Constructor.

            :param input_dict: input dictionary.
        """
        self.__dict__ = input_dict


def if_none(input_value, default_value):
    """Return the original value if not None and the default value otherwise.

    :param input_value: input value
    :param default_value: default value
    :return: the input value or default (if the input value is None)
    """
    return input_value if input_value is not None else default_value
