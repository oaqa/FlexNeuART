import random
from abc import abstractmethod
from flexneuart.data_augmentation.utils.abnirml_transformation import *

class DataAugmentModule:
    def __init__(self, random_seed=42):
        self.random_seed = random.Random(random_seed)
        
    @abstractmethod
    def augment(self, query_text, doc_text):
        pass


class RandomDataAugmentModule(DataAugmentModule):
    def __init__(self, random_seed=42):
        super().__init__(random_seed)
        self.doc_augment = ShufSents()

    def augment(self, query_text, doc_text):
        return query_text, self.doc_augment.augment(doc_text)
