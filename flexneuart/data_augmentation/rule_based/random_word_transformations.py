from flexneuart.data_augmentation.rule_based.data_augment import DataAugment
from flexneuart.data_augmentation import register_augmentation
import re
import random
import json


# This file contains functions that would do augmentatation on random words in the document

@register_augmentation("random_word_insertion")
class RandomWordInsertion(DataAugment):
    """
    A class used for Inserting Random words in the Document
    ...

    Attributes
    ----------
    alpha_ri : float
        Percentage of random words to insert

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.alpha_ri =  conf[self.augmentation_name]["probability"]
            self.types = conf[self.augmentation_name].get("types", self.types)
        except:
            expected_config = {self.augmentation_name :
                               {"alpha_ri": 0.1}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))

    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        number_ri = max(1, int(self.alpha_ri*num_words))
        augmented_words = self.__random_insertion(words, number_ri)
        return ' '.join(augmented_words)
    
    def __random_insertion(self, words, n):
        """
        This method calls the add_word method every time we want of add a word

        Parameters:
        ----------
        words : List[str]
            The list of all words from the document
        n : int
            The number of words to be inserted in the document
        """
        new_words = words.copy()
        for _ in range(n):
            self.__add_word(new_words)
        return new_words
        
    def __add_word(self, new_words):
        """
        Sample a word from a list of words and add it to a random place

        Parameters:
        ----------
        new_words : List[str]
            A list of words to sample a single word from
        """
        repeat_word = new_words[random.randint(0, len(new_words)-1)]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, repeat_word)

        
@register_augmentation("random_word_deletion")
class RandomWordDeletion(DataAugment):
    """
    A class used for Inserting Random words in the Document
    ...

    Attributes
    ----------
    p : float
        The probability of deleting every word in the document

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.p = conf[self.augmentation_name]["probability"]
            self.types = conf[self.augmentation_name].get("types", self.types)
        except:
            expected_config = {self.augmentation_name :
                               {"p": 0.1}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))

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


@register_augmentation("random_word_swap")
class RandomWordSwap(DataAugment):
    """
    A class used for Inserting Random words in the Document
    ...

    Attributes
    ----------
    alpha_rs : float
        Percentage of the document that would be swapped

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.alpha_rs = conf[self.augmentation_name]["probability"]
            self.types = conf[self.augmentation_name].get("types", self.types)
        except:
            expected_config = {self.augmentation_name :
                               {"alpha_rs": 0.1}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))

    def augment(self, text):
        words = re.split('\s+', text)
        num_words = len(words)
        number_rs = max(1, int(self.alpha_rs*num_words))
        augmented_words = self.__random_swap(words, number_rs)
        return ' '.join(augmented_words)

    def __random_swap(self, words, n):
        """
        Wrapper method to call swap_words function

        Parameters:
        ----------
        words : List[str]
            A list of all words from the document
        n : int
            The number of words to swap
        """
        new_words = words.copy()
        for _ in range(n):
            new_words = self.__swap_word(new_words)
        return new_words

    def __swap_word(self, new_words):
        """
        Swap a word

        Parameters:
        ----------
        new_words : List[str]
            A list of words where 2 indexes will be chosen to swap the words
        """
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_words
