from base_class import DataAugment
import re
class RandomInsertion(DataAugment):
    def augment(self, text, **kwargs):
        alpha_ri = 0.1 # percentage of random insertion
        words = re.split('\s+', text)
        num_words = len(words)
        number_ri = max(1, int(alpha_ri*num_words))
		a_words = self.random_insertion(words, n_ri)
			augmented_sentences.append(' '.join(a_words))


class RandomDeletion(DataAugment):
    def augment(text, **kwargs):
        pass

class RandomSwap(DataAugment):
    def augment(text, **kwargs):
        pass