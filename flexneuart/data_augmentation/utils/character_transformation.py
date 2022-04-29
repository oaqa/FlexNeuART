from base_class import DataAugment
import re
import string
import random

class AddCharacterTransformation(DataAugment):
    def __init__(self, word_add_probability=0.1, character_add_probability=0.1, random_seed=42):
        super().__init__(random_seed)
        self.word_add_probability = word_add_probability
        self.character_add_probability = character_add_probability

    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        for i in range(num_words):
            r = random.uniform(0, 1)
            if r > self.word_add_probability:
                words[i] = self.add_letters_with_probability(words[i])
        return ' '.join(words)

    def add_single_letter(self, word):
        # adds a single letter to the word at random index
        k = len(word)
        i = randint(1, k-1)
        word = word[:i] + random.choice(string.ascii_lowercase) + word[i:]
        return word
    
    def add_letters_with_probability(self, word):
        # adds letters to the word at random index, based on probability
        k = len(word)
        if k <= 1:
            return word
        i = 1
        while i < k - 1:
            if random.random() < self.character_add_probability:
                word = word[:i] + random.choice(string.ascii_lowercase) + word[i:]
            i += 1
        return word

class RemoveCharacterTransformation(DataAugment):
    def __init__(self, word_remove_character_probability=0.1, character_remove_probability=0.1, random_seed=42):
        super().__init__(random_seed)
        self.word_remove_character_probability = word_remove_character_probability
        self.character_remove_probability = character_remove_probability

    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        for i in range(num_words):
            r = random.uniform(0, 1)
            if r > self.word_remove_character_probability:
                words[i] = self.remove_letters_with_probability(words[i])
        return ' '.join(words)

    def remove_single_letter(self, word):
        # adds a single letter to the word at random index
        k = len(word)
        i = randint(1, k-1)
        word = word[:i] + " " + word[i+1:]
        word = re.sub(" ", "", word)
        return word
    
    def remove_letters_with_probability(self, word):
        # adds letters to the word at random index, based on probability
        k = len(word)
        if k<=1:
            return word
        i = 1
        while i < k - 1:
            if random.random() < self.character_remove_probability:
                word = word[:i] + " " + word[i+1:]
            i += 1
        word = re.sub(" ", "", word)
        return word

class SwapCharacterTransformation(DataAugment):
    def __init__(self, word_level_probability=0.1, character_swap_probability=0.1, random_seed=42):
        super().__init__(random_seed)
        self.word_level_probability = word_level_probability
        self.character_swap_probability = character_swap_probability

    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        for i in range(num_words):
            r = random.uniform(0, 1)
            if r > self.word_level_probability:
                words[i] = self.swap_letters_with_probability(words[i])
        return ' '.join(words)
    
    def swap_letters_with_probability(self, word):
        # adds letters to the word at random index, based on probability
        k = len(word)
        if k < 2:
            return word
        else:
            i = 0
            skip = False
            for i in range(k-2):
                if skip == False and random.random() < self.character_swap_probability:
                    word = word[:i] + word[i+1] + word[i] + word[i+2:]
                    skip = True
                else:
                    skip = False
            if skip == False and random.random() < self.character_swap_probability:
                word = word[:k-2] + word[k-1] + word[k-2]
        return word

