from base_class import DataAugment
import re
import random
from nltk.corpus import wordnet 

class RandomWordInsertion(DataAugment):
    def __init__(self, alpha_ri=0.1, random_seed=42):
        super().__init__(random_seed)
        self.alpha_ri = alpha_ri # percentage of random insertion

    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        number_ri = max(1, int(self.alpha_ri*num_words))
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

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)


class RandomWordDeletion(DataAugment):
    def __init__(self, p = 0.2, random_seed=42):
        super().__init__(random_seed)
        self.p = p


    def augment(self, text):
        # Code referenced from: https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py

        words = text.split()

        # No deletion if only one word is present
        if len(words) == 1:
            return words

        # Delete words randomly with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > self.p:
                new_words.append(word)

        # Return a random word if all words deleted
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]

        return " ".join(new_words)


class RandomWordSwap(DataAugment):
    def __init__(self, alpha_rs=0.1, random_seed=42):
        super().__init__(random_seed)
        self.alpha_rs = alpha_rs # percentage of random swap

    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        number_rs = max(1, int(self.alpha_rs*num_words))
        augmented_words = self.random_swap(words, number_rs)
        return ' '.join(augmented_words)

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_wordsOB
