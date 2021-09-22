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
    This processor just passes a given number of fields to the next stage
    (without any modifications).
"""

from flexneuart.ir_datasets.base import BaseTextProcessor
from flexneuart.ir_datasets import register


@register('copy')
class CopyTextProcessor(BaseTextProcessor):

    def __init__(self, input_fields : list):
        """Constructor.

            :param input_fields:     a list of field names to copy
        """
        self.input_fields = input_fields

    def __call__(self, input_dict: dict):
        output = {field_name : field_val
                  for field_name, field_val in input_dict.items()
                  if field_name in self.input_fields}

        return output
