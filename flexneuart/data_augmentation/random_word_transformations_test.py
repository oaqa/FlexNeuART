from random_word_transformations import RandomDeletion

class TestRandomWordTransformations():
    def __init__(self):
        pass

    def test_random_deletions(self):
        text = "Testing random deletions in this sentence. Do you think words are correctly deleted"
        random_deletion = RandomDeletion(p = 0.5)
        text_words_deleted = random_deletion.augment(text)
        assert(len(text_words_deleted) < len(text))

if __name__ == "__main__":
    tester = TestRandomWordTransformations()
    tester.test_random_deletions()
