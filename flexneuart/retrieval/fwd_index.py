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
Access to FlexNeuART forward index.
"""
from typing import Union
from collections import namedtuple

DocEntryParsed=namedtuple('DocEntryParsed',
                          ['word_ids', # a list of unique word IDs
                           'word_qtys', # a list of the # of occurrences of unique words
                           'word_id_seq', # a sequence of word IDs as they appear in a document at index time (may be
                                          # missing stop words). It is None for non-positional indices
                           'doc_len', # document length in the number of words (at index time)
                           ])

WordEntry=namedtuple('WordEntry',
                     ['word_id',   # word ID
                      'word_freq'  # of documents containing at least one of these words
                      ])

class ForwardIndex:
    def __init__(self, resource_manager, field_name):
        """Constructor of the forward index wrapper object.

        :param resource_manager:    a resource manager reference.
        :param field_name:          the name of the field, e.g., text
        """
        self.field_name = field_name
        self.indx = resource_manager.getFwdIndex(field_name)
        self.indx_fld_type = self.indx.getIndexFieldType()

    def check_is_text_raw(self):
        if not self.indx.isTextRaw():
            raise Exception(f'Field {self.field_name} is not raw text but {self.indx_fld_type}')

    def check_is_parsed(self):
        if not self.indx.isParsed():
            raise Exception(f'Field {self.field_name} is not parsed but {self.indx_fld_type}')

    def check_is_parsed_text(self):
        if not self.indx.isParsedText():
            raise Exception(f'Field {self.field_name} is not parsed (with positional info) but {self.indx_fld_type}')

    def check_is_binary(self):
        if not self.indx.isBinary():
            raise Exception(f'Field {self.field_name} is not binary but {self.indx_fld_type}')

    def get_doc_raw(self, doc_id):
        """For backwards compatibility only."""
        return self.get_doc_text_raw(doc_id)

    def get_doc_text_raw(self, doc_id):
        """Obtain the raw-document text. Must be a raw-field.

            :param doc_id: a document ID (e.g., returned by a candidate provider)
            :return:   document text or None if no such document exists
        """
        self.check_is_text_raw()

        return self.indx.getDocEntryTextRaw(doc_id)


    def get_doc_entry_parsed_text(self, doc_id):
        """Retrieves an existing parsed document entry and constructs a textual representation.
           This function needs a positional forward index.

            :param doc_id: a document ID (e.g., returned by a candidate provider)
            :return:   document text or None if no such document exists
        """
        self.check_is_parsed_text()

        return self.indx.getDocEntryParsedText(doc_id)

    def get_doc_text(self, doc_id):
        """Retrieves a document text. For raw text, it just retrieves the original text, for parsed texxt
           fields, the text is reconstructed.

           :param doc_id: a document ID (e.g., returned by a candidate provider)
           :return:   document text or None if no such document exists
        """
        if self.indx.isTextRaw():
            return self.indx.getDocEntryTextRaw(doc_id)
        elif self.indx.isParsedText():
            return self.indx.getDocEntryParsedText(doc_id)
        else:
            raise Exception(f'Unsupported field type: {self.indx_fld_type}')

    def get_doc_parsed(self, doc_id):
        """Get a parsed document entry.

        :param doc_id:  a document ID (e.g., returned by a candidate provider)
        :return:        an object of the type DocEntryParsed
        """
        self.check_is_parsed()

        entry = self.indx.getDocEntryParsed(doc_id)
        if entry is None:
            return None

        return DocEntryParsed(word_ids=entry.mWordIds, word_qtys=entry.mQtys,
                              word_id_seq=entry.mWordIdSeq, doc_len=entry.mDocLen)

    def get_word_entry_by_id(self, word_id) -> Union[WordEntry,None]:
        """Retrieve word entry/info by word ID

        :param word_id:  an integer word ID
        :return: an object of the type WordEntry or None
        """
        self.check_is_parsed()
        assert type(word_id) == int, "word_id must be integer!"

        entry = self.indx.getWordEntry(word_id)
        if entry is None:
            return None

        return WordEntry(word_id=entry.mWordId, word_freq=entry.mWordFreq)

    def get_word_by_id(self, word_id):
        """Retrieve the word string by ID.

        :param word_id:  an integer word ID
        :return: a word or None if ID does not exist
        """
        self.check_is_parsed()
        assert type(word_id) == int, "word_id must be integer!"

        return self.indx.getWord(word_id)

    def get_all_word_ids(self):
        """Retrieve all word IDs."""
        return self.indx.getAllWordIds()

    def get_all_doc_ids(self):
        """Retrieve all document IDs"""
        return self.indx.getAllDocIds()

    def get_doc_qty(self):
        """
        :return: a number of documents in the index.
        """
        return self.indx.getDocQty()

    def get_avg_doc_len(self):
        """
        :return: the average document length in the number of tokens (it makes sense only for parsed indices)
        """
        self.check_is_parsed()
        return self.indx.getAvgDocLen()


def get_forward_index(resource_manager, field_name):
    """Create a wrapper for a forward index class.

    :param resource_manager:    a resource manager reference.
    :param field_name:          the name of the field, e.g., text

    :return: an object of the type ForwardIndex. There will be an exception if the index is not present.
    """
    return ForwardIndex(resource_manager, field_name)

