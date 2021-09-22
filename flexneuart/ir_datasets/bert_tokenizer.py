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
    This processor carries out BERT tokenization.
"""

from flexneuart.ir_datasets import register
from flexneuart.ir_datasets.base import BaseTextProcessor
from flexneuart.text_proc.parse import get_bert_tokenizer, get_retokenized

@register('bert_tokenizer')
class BertTokenizerProcessor(BaseTextProcessor):

    def __init__(self, input_fields : list):
        """Constructor.

        :param input_fields:     a list of field names to process
        """
        self.input_fields = frozenset(input_fields)

        self.tokenizer = get_bert_tokenizer()

    def __call__(self, input_dict: dict):
        """BERT-tokenizes each input field whose name was specified in the constructor.
           The output fields are obtained by suffixing original input field names
           with '.bert_tokens'

        :param input_dict:
        :return:
        """
        output_dict ={}

        for field_name, field_val in input_dict.items():
            if field_name in self.input_fields:
                output_dict[field_name + '.bert_tokens'] = get_retokenized(self.tokenizer, field_val)

        return output_dict