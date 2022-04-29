import spacy
import random


class DataAugment:
    def __init__(self, spacy_model='en_core_web_sm', random_seed=42):
        self.nlp = spacy.load(spacy_model)
        self.random = random.Random(random_seed)

        return

    @abstractmethod
    def augment(text, **kwargs):
        pass       
