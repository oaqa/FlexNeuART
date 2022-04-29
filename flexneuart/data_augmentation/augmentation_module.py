import random

class DataAugmentModule:
    def __init__(self, random_seed=42):
        self.random_seed = random.Random(random_seed)

    @abstractmethod
    def augment(self, query_text, doc_text):
        pass


class RandomDataAugmentModule(DataAugmentionModule):
    def __init__(self, random_seed=42):
        super().__init__(random_seed)
        self.query_augment = None
        self.doc_augment = None

    def augment(self, query_text, doc_text):
        return self.query_augment(query_text), self.doc_augment(doc_text)
