"""
Access to FlexNeuART forward index.
"""
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
        self.is_raw = self.indx.isRaw()


    def get_doc_raw(self, doc_id):
        """Obtain the raw-document text. Must be a raw-field.

        :param doc_id: a document ID (e.g., returned by a candidate provider)
        :return:   document text or None if no such document exists
        """
        self.check_raw_or_not(check_raw=True)

        return self.indx.getDocEntryRaw(doc_id)

    def get_word_entry_by_id(self, word_id):
        """Retrieve word entry/info by word ID

        :param word_id:  an integer word ID
        :return: an object of the type WordEntry or None
        """
        self.check_raw_or_not(check_raw=False)
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
        self.check_raw_or_not(check_raw=False)
        assert type(word_id) == int, "word_id must be integer!"

        return self.indx.getWord(word_id)

    def get_doc_parsed(self, doc_id):
        """Get a parsed document entry.

        :param doc_id:  a document ID (e.g., returned by a candidate provider)
        :return:        an object of the type DocEntryParsed
        """
        self.check_raw_or_not(check_raw=False)

        entry = self.indx.getDocEntryParsed(doc_id)
        if entry is None:
            return None

        return DocEntryParsed(word_ids=entry.mWordIds, word_qtys=entry.mQtys,
                              word_id_seq=entry.mWordIdSeq, doc_len=entry.mDocLen)

    def check_raw_or_not(self, check_raw):
        if check_raw:
            if not self.is_raw:
                raise Exception(f'Field {self.field_name} is parsed and not raw text!')
        else:
            if self.is_raw:
                raise Exception(f'Field {self.field_name} is raw text rather than parsed documents!')


def get_forward_index(resource_manager, field_name):
    """Create a wrapper for a forward index class.

    :param resource_manager:    a resource manager reference.
    :param field_name:          the name of the field, e.g., text

    :return: an object of the type ForwardIndex. There will be an exception if the index is not present.
    """
    return ForwardIndex(resource_manager, field_name)

