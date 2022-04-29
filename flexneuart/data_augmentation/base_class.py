import random

class DataAugment:
    def __init__(self, random_seed=42):
        self.random_seed = random.Random(random_seed)
        return

    @abstractmethod
    def augment(self, text):
        pass       
