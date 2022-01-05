"""
    This processor is a simple (basically a white space tokenizer) which
    applies the Krovetz stemmer to tokens.
"""

from flexneuart.text_proc.parse import KrovetzStemParser
from flexneuart.io.stopwords import read_stop_words

from flexneuart.ir_datasets.base import BaseTextProcessor
from flexneuart.ir_datasets import register


@register('krovetz_stemmer')
class KrovetzStemProcessor(BaseTextProcessor):

    def __init__(self, input_fields : list,
                 stop_word_file=None):
        """Constructor.

        :param input_fields:   a list of field names to process
        :param stop_word_file  the name of the stop word file
        """
        stop_words = read_stop_words(stop_word_file) if stop_word_file is not None else []

        self.input_fields = frozenset(input_fields)

        self.parser = KrovetzStemParser(stop_words)

    def __call__(self, input_dict: dict):
        """Parses each input field whose name was specified in the constructor.
           For each field it extracts tokens and stems them. The output field
           name is obtained by prefixing the '.stems' with the
           original input field name.

        :param input_dict:
        :return:
        """
        output_dict ={}

        for field_name, field_val in input_dict.items():
            if field_name in self.input_fields:
                output_dict[field_name + '.stems'] = self.parser(field_val)

        return output_dict
