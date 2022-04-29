
class DataAugment:
    def __init__(self):
        return

    @abstractmethod
    def augment(self, text, **kwargs):
        pass       
