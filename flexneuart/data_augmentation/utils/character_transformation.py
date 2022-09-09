from flexneuart.data_augmentation.utils.base_class import DataAugment
import re
import string
import random

KEYBOARD_POSITIONS = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
                      ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
                      ['z', 'x', 'c', 'v', 'b', 'n', 'm']]


class AddCharacterTransformation(DataAugment):
    """
    A class used for character addition data augmentation

    ...

    Attributes
    ----------
    word_add_probability : float
        the probability with which to select words to modify
    character_add_probability : float
        the probability with which to add characters within the selected word

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    
    def __init__(self, word_add_probability=0.1, character_add_probability=0.1):
        """
        Parameters
        ----------
        word_add_probability : float
            the probability with which to select words to modify
        character_add_probability : float
            the probability with which to add characters within the selected word
        """
        super().__init__()
        self.word_add_probability = word_add_probability
        self.character_add_probability = character_add_probability

    def augment(self, text):
        """
        Returns the augmented text
        
        Parameters
        ----------
        text : str
            text to be augmented
        """
        words = re.split('\s+', text)
        for i in range(len(words)):
            if random.random() < self.word_add_probability:
                words[i] = self.__add_letters_with_probability(words[i])
        return ' '.join(words)
    
    def __add_letters_with_probability(self, word):
        """
        Adds characters to the word based on character add probability

        Parameters
        ----------
        word : str
            word to add characters to
        """
        for i in range(len(word) - 1, 0, -1):
            if random.random() < self.character_add_probability:
                word = word[:i] + random.choice(string.ascii_lowercase) + word[i:]
        return word


class RemoveCharacterTransformation(DataAugment):
    """
    A class used for character deletion data augmentation

    ...

    Attributes
    ----------
    word_remove_probability : float
        the probability with which to select words to modify
    character_remove_probability : float
        the probability with which to delete characters within the selected word

    Methods
    -------
    augment(text)
        returns the augmented text
    """

    def __init__(self, word_remove_probability=0.1, character_remove_probability=0.1):
        """
        Parameters
        ----------
        word_remove_probability : float
            the probability with which to select words to modify
        character_remove_probability : float
            the probability with which to delete characters within the selected word
        """
        super().__init__()
        self.word_remove_probability = word_remove_probability
        self.character_remove_probability = character_remove_probability

    def augment(self, text):
        """
        Returns the augmented text
        
        Parameters
        ----------
        text : str
            text to be augmented
        """
        words = re.split('\s+', text)
        for i in range(len(words)):
            if random.random() < self.word_remove_probability:
                words[i] = self.__remove_letters_with_probability(words[i])
        return ' '.join(words)

    def __remove_letters_with_probability(self, word):
        """
        Removes characters in the word based on character remove probability

        Parameters
        ----------
        word : str
            word to delete characters to
        """
        for i in range(len(word) - 2, 0, -1):
            if random.random() < self.character_remove_probability:
                word = word[:i] + word[i+1:]
        return word


class SwapCharacterTransformation(DataAugment):
    """
    A class used for adjacent character swap data augmentation

    ...

    Attributes
    ----------
    word_swap_probability : float
        the probability with which to select words to swap
    character_swap_probability : float
        the probability with which to swap characters within the selected word

    Methods
    -------
    augment(text)
        returns the augmented text
    """

    def __init__(self, word_swap_probability=0.1, character_swap_probability=0.1):
        """
        Parameters
        ----------
        word_swap_probability : float
            the probability with which to select words to swap
        character_swap_probability : float
            the probability with which to swap characters within the selected word
        """
        super().__init__()
        self.word_swap_probability = word_swap_probability
        self.character_swap_probability = character_swap_probability

    def augment(self, text):
        """
        Returns the augmented text
        
        Parameters
        ----------
        text : str
            text to be augmented
        """
        words = re.split('\s+', text)
        for i in range(len(words)):
            if random.random() < self.word_swap_probability:
                words[i] = self.__swap_letters_with_probability(words[i])
        return ' '.join(words)
    
    def __swap_letters_with_probability(self, word):
        """
        Swaps adjacent characters in the word based on character swap probability
        
        Parameters
        ----------
        word : str
            word to swap characters within
        """
        skip = False
        for i in range(1, len(word) - 2):
            if skip == False and random.random() < self.character_swap_probability:
                word = word[:i] + word[i+1] + word[i] + word[i+2:]
                skip = True
            else:
                skip = False
        return word

    
class ReplaceCharacterTransformation(DataAugment):
    """
    A class used for replacing characters data augmentation

    ...

    Attributes
    ----------
    word_replace_probability : float
        the probability with which to select words to modify
    character_replace_probability : float
        the probability with which to replace characters within the selected word

    Methods
    -------
    augment(text)
        returns the augmented text
    """

    def __init__(self, word_replace_probability=0.1, character_replace_probability=0.1):
        """
        Parameters
        ----------
        word_replace_probability : float
            the probability with which to select words to modify
        character_replace_probability : float
            the probability with which to replace characters within the selected word
        """
        super().__init__()
        self.word_replace_probability = word_replace_probability
        self.character_replace_probability = character_replace_probability
        self.keyboard_positions = KEYBOARD_POSITIONS

    def augment(self, text):
        """
        Returns the augmented text
        
        Parameters
        ----------
        text : str
            text to be augmented
        """
        words = re.split('\s+', text)
        for i in range(len(words)):
            if random.random() < self.word_replace_probability:
                words[i] = self.__replace_letters_with_probability(words[i])
        return ' '.join(words)

    def __replace_letters_with_probability(self, word):
        """
        Replace character with character replacement probability

        Parameters
        ----------
        word : str
            word to be modified
        """
        for i in range(1, len(word) - 2):
            if random.random() < self.character_replace_probability:
                word = word[:i] + random.choice(string.ascii_lowercase) + word[i+1:]
        return word


class ReplaceCharacterKeyboardTransformation(DataAugment):
    """
    A class used for replacing characters with nearby keyboard characters data augmentation

    ...

    Attributes
    ----------
    word_replace_probability : float
        the probability with which to select words to modify
    character_replace_probability : float
        the probability with which to replace characters within the selected word

    Methods
    -------
    augment(text)
        returns the augmented text
    """

    def __init__(self, word_replace_probability=0.1, character_replace_probability=0.1):
        """
        Parameters
        ----------
        word_replace_probability : float
            the probability with which to select words to modify
        character_replace_probability : float
            the probability with which to replace characters within the selected word
        """
        super().__init__()
        self.word_replace_probability = word_replace_probability
        self.character_replace_probability = character_replace_probability
        self.keyboard_positions = KEYBOARD_POSITIONS

    def augment(self, text):
        """
        Returns the augmented text
        
        Parameters
        ----------
        text : str
            text to be augmented
        """
        words = re.split('\s+', text)
        for i in range(len(words)):
            if random.random() < self.word_replace_probability:
                words[i] = self.__replace_letters_with_probability(words[i])
        return ' '.join(words)

    def __get_closest_from_keyboard(self, character):
        """
        Returns the closest keyboard character

        Parameters
        ----------
        character : str
            character based on which the closest keyboard character is chosen
        """
        pos = (0, 0)
        return_pos = (0, 0)
        for i, x in enumerate(self.keyboard_positions):
            if character in x:
                pos = (i, x.index(character))
        i = pos[0]
        j = pos[1]
        if j == 0:
            return_pos = (i, j + 1)
        elif j == len(self.keyboard_positions[i])-1:
            return_pos = (i, j - 1)
        else:
            r = random.uniform(0, 1)
            if r > 0.5:
                return_pos = (i, j + 1)
            else:
                return_pos = (i, j - 1)
        return self.keyboard_positions[return_pos[0]][return_pos[1]]

    def __replace_letters_with_probability(self, word):
        """
        Replace character with the closest keyboard character with character replacement probability

        Parameters
        ----------
        word : str
            word to be modified
        """
        for i in range(1, len(word) - 2):
            if random.random() < self.character_replace_probability:
                word = word[:i] + self.__get_closest_from_keyboard(word[i]) + word[i+1:]
        return word


class AddCharacterKeyboardAdjacentTransformation(DataAugment):
    """
    A class used for adding nearby keyboard characters data augmentation

    ...

    Attributes
    ----------
    word_add_probability : float
        the probability with which to select words to modify
    character_add_probability : float
        the probability with which to add characters within the selected word

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    
    def __init__(self, word_add_probability=0.1, character_add_probability=0.1):
        """
        Parameters
        ----------
        word_add_probability : float
            the probability with which to select words to modify
        character_add_probability : float
            the probability with which to add characters within the selected word
        """
        super().__init__()
        self.word_add_probability = word_add_probability
        self.character_add_probability = character_add_probability
        self.keyboard_positions = KEYBOARD_POSITIONS


    def augment(self, text):
        """
        Returns the augmented text
        
        Parameters
        ----------
        text : str
            text to be augmented
        """
        words = re.split('\s+', text)
        for i in range(len(words)):
            if random.random() < self.word_add_probability:
                words[i] = self.__add_letters_with_probability(words[i])
        return ' '.join(words)

    def __get_closest_from_keyboard(self, character):
        """
        Returns the closest keyboard character

        Parameters
        ----------
        character : str
            character based on which the closest keyboard character is chosen
        """
        pos = (0, 0)
        return_pos = (0, 0)
        for i, x in enumerate(self.keyboard_positions):
            if character in x:
                pos = (i, x.index(character))
        i = pos[0]
        j = pos[1]
        if j == 0:
            return_pos = (i, j + 1)
        elif j == len(self.keyboard_positions[i])-1:
            return_pos = (i, j - 1)
        else:
            r = random.uniform(0, 1)
            if r > 0.5:
                return_pos = (i, j + 1)
            else:
                return_pos = (i, j - 1)
        return self.keyboard_positions[return_pos[0]][return_pos[1]]
    
    def __add_letters_with_probability(self, word):
        """
        Add closest keyboard character with character add probability

        Parameters
        ----------
        word : str
            word to be modified
        """
        for i in range(len(word) - 2, 0, -1):
            if random.random() < self.character_add_probability:
                word = word[:i] + self.__get_closest_from_keyboard(i) + word[i:]
        return word
