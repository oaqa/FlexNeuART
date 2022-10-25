from abc import abstractmethod

class DataAugment:
    def __init__(self):
        return
        
    @abstractmethod
    def augment(self, text, query=None):
        pass       
