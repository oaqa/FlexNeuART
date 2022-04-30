import random
from abc import abstractmethod
from flexneuart.data_augmentation.utils.abnirml_transformation import *
from flexneuart.data_augmentation.utils.random_word_transformations import *
from flexneuart.data_augmentation.utils.synonym_hypernym_transformations import *
from flexneuart.data_augmentation.utils.document_level_transformation import *

class DataAugmentModule:
    def __init__(self, augment_type):
        self.doc_augment = None
        if augment_type == 'random_word_deletion':
            self.doc_augment = RandomWordDeletion(p = 0.05)
        elif augment_type == 'random_word_insertion':
            self.doc_augment = RandomWordInsertion(alpha_ri = 0.05)
        # elif augment_type == 'constant_document_length':
        #     self.doc_augment = ConstantDocLength(doc_length=500) # doc length needs to be decided
        
    def augment(self, query_text, doc_text):
        if self.doc_augment is not None:
            doc_text = self.doc_augment.augment(doc_text)
        return query_text, doc_text
