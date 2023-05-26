from flexneuart.data_augmentation.rule_based.random_word_transformations import RandomWordDeletion, RandomWordInsertion, RandomWordSwap
from test_conf import conf

class TestRandomWordTransformations():
    def test_random_swap(self):
        sentence = "This is a test sentence"
        random_swap_sentence = RandomWordSwap("random_word_swap", conf)
        print(random_swap_sentence.augment(sentence))
    
    def test_random_insertions(self):
        sentence = "This is a test sentence"
        random_inserted_sentence = RandomWordInsertion("random_word_insertion", conf)
        print(random_inserted_sentence.augment(sentence))

    def test_random_deletions(self):
        text = "Testing random deletions in this sentence. Do you think words are correctly deleted"
        random_deletion = RandomWordDeletion("random_word_deletion", conf)
        text_words_deleted = random_deletion.augment(text)
        assert(len(text_words_deleted) < len(text))

if __name__ == "__main__":
    tester = TestRandomWordTransformations()
    tester.test_random_deletions()
    tester.test_random_swap()
    tester.test_random_insertions()
