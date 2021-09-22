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
    This processor just renames a given set of fields. It can be used to
    copy (pass through) a given list of attributes.
"""

from flexneuart.ir_datasets.base import BaseTextProcessor
from flexneuart.ir_datasets import register

@register('rename')
class RenameTextProcessor(BaseTextProcessor):

    def __init__(self, rename_dict : dict):
        """Constructor.

            :param rename_dict: a key is the original name and the value is the
        """
        self.rename_dict = rename_dict

    def __call__(self, input_dict: dict):
        output_dict ={}

        for src_name, val in input_dict.items():
            if src_name in self.rename_dict:
                output_dict[self.rename_dict[src_name]] = val

        return output_dict
