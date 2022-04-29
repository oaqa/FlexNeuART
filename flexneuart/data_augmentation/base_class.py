import random
from abc import abstractmethod
class DataAugment:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

        return

    @abstractmethod
    def augment(self, text):
        pass       
