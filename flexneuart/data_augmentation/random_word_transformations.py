from base_class import DataAugment
import re
class RandomInsertion(DataAugment):
    def augment(self, text, **kwargs):
        alpha_ri = 0.1 # percentage of random insertion
        words = re.split('\s+', text)
        num_words = len(words)
        number_ri = max(1, int(alpha_ri*num_words))
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))


class RandomDeletion(DataAugment):
    def __init__(p = 0.2):
        self.p = p
        super().__init__()

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


class RandomSwap(DataAugment):
    def augment(self, text, **kwargs):
        pass