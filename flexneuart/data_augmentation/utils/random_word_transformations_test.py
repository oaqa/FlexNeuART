from random_word_transformations import RandomWordDeletion, RandomWordInsertion, RandomWordSwap

class TestRandomWordTransformations():
    def test_random_swap(self):
        sentence = "This is a test sentence"
        random_swap_sentence = RandomWordSwap(alpha_rs = 0.2)
        print(random_swap_sentence.augment(sentence))
    
    def test_random_insertions(self):
        sentence = "This is a test sentence"
        random_inserted_sentence = RandomWordInsertion(alpha_ri = 0.2)
        print(random_inserted_sentence.augment(sentence))
    
    def test_random_deletions(self):
        sentence = "This is a test sentence"
        random_deletion = RandomWordDeletion(p = 0.1)
        print(random_deletion.augment(sentence))

    def test_random_deletions(self):
        text = "Testing random deletions in this sentence. Do you think words are correctly deleted"
        random_deletion = RandomWordDeletion(p = 0.5)
        text_words_deleted = random_deletion.augment(text)
        assert(len(text_words_deleted) < len(text))

if __name__ == "__main__":
    tester = TestRandomWordTransformations()
    tester.test_random_deletions()
    tester.test_random_swap()
    tester.test_random_insertions()
