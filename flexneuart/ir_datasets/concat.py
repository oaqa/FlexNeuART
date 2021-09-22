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
    This processor just concatenates fields. If a field value is missing in the input
    dictionary, we default to using an empty string.
"""

from flexneuart.ir_datasets.base import BaseTextProcessor
from flexneuart.ir_datasets import register


@register('concat')
class ConcatTextProcessor(BaseTextProcessor):

    def __init__(self, input_fields : list, output_field : str):
        """Constructor.

            :param input_fields:     a list of field names to concatenate (order matters)
            :param output_field:     the name of the output fields
        """
        self.input_fields = input_fields
        self.output_field = output_field

    def __call__(self, input_dict: dict):
        output_parts = []
        for field_name in self.input_fields:
            val = input_dict.get(field_name)
            output_parts.append(val if val is not None else '')

        return {self.output_field : ' '.join(output_parts)}
