from base_class import DataAugment
import re
import random
class RandomInsertion(DataAugment):
    def __init__(self, alpha_ri=0.1, random_seed=42):
        super().__init__(random_seed)
        self.alpha_ri = alpha_ri # percentage of random insertion
        
    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        number_ri = max(1, int(alpha_ri*num_words))
        augmented_words = self.random_insertion(words, number_ri)
        return ' '.join(augmented_words)
        
    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words
        
    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)


class RandomDeletion(DataAugment):
    def augment(text, **kwargs):
        pass

class RandomSwap(DataAugment):
    def augment(text, **kwargs):
        pass