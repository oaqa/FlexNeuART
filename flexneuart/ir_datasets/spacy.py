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
    This processor tokenizes input and optionally removes the stop words.
"""

from flexneuart.text_proc.parse import SpacyTextParser
from flexneuart.io.stopwords import read_stop_words

from flexneuart.ir_datasets.base import BaseTextProcessor
from flexneuart.ir_datasets import register

@register('spacy')
class SpacyTextProcessor(BaseTextProcessor):

    def __init__(self, input_fields : list,
                 model_name, stop_word_file=None,
                 remove_punct=True,
                 keep_only_alpha_num=False,
                 lower_case=True,
                 enable_pos=True):
        """Constructor.

        :param input_fields:   a list of field names to process
        :param  model_name     a name of the spacy model to use, e.g., en_core_web_sm
        :param  stop_word_file  the name of the stop word file
        :param  remove_punct  a bool flag indicating if the punctuation tokens need to be removed
        :param  keep_only_alpha_num a bool flag indicating if we need to keep only alpha-numeric characters
        :param  enable_pos    a bool flag that enables POS tagging (which, e.g., can improve lemmatization)
        """
        stop_words = read_stop_words(stop_word_file) if stop_word_file is not None else []

        self.input_fields = frozenset(input_fields)

        self.nlp = SpacyTextParser(model_name,
                                   stop_words=stop_words,
                                   remove_punct=remove_punct,
                                   sent_split=False,
                                   keep_only_alpha_num=keep_only_alpha_num,
                                   lower_case=lower_case,
                                   enable_pos=enable_pos
                                   )

    def __call__(self, input_dict: dict):
        """Parses each input field whose name was specified in the constructor.
           For each field it extracts tokens and lemmas. The output fields
           are obtained by prefixing strings '.lemmas' and '.tokens' with the
           original input field name.

        :param input_dict:
        :return:
        """
        output_dict ={}

        for field_name, field_val in input_dict.items():
            if field_name in self.input_fields:
                lemmas, tokens = self.nlp.proc_text(field_val)
                output_dict[field_name + '.lemmas'] = lemmas
                output_dict[field_name + '.tokens'] = tokens

        return output_dict
