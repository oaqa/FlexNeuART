from base_class import DataAugment
import re
import random
from nltk.corpus import wordnet 

class ConstantDocLength(DataAugment):
    def __init__(self, doc_length, random_seed=42):
        super().__init__(random_seed)
        self.doc_length = doc_length # maximum tokens needed of a doc

    def augment(self, text):
        tokens = text.split()
        old_length = len(tokens)
        constant_length_text = []

        if old_length>self.doc_length:
            indices_to_delete = random.sample(range(old_length),old_length-self.doc_length)

            for ind,word in enumerate(tokens):
                if ind not in indices_to_delete:
                    constant_length_text.append(word)           
        
        return " ".join(constant_length_text)
