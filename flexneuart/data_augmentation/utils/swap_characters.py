from base_class import DataAugment
import spacy

class SwapCharacters:
    def __init__(self, **kwargs):
        super().__init__()
        self.prob = kwargs.get('char_swap_probability', 0.1)
        self.nlp = spacy.load()
    
    def augment(self, text):
        tokens = self.nlp(text)
        num_tokens = len(tokens)

        
