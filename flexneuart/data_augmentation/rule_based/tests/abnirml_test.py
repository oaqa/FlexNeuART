import unittest
from abnirml_transformation import *

class TestSimpleaugmentations(unittest.TestCase):

    def test_casefold(self):
        to_lower = CaseFold()
        query = "AppLE"
        lower_query = to_lower.augment(query)

        self.assertEqual(lower_query, "apple")
    
    def test_delete_punctuation(self):
        del_punct = DelPunct()
        query = "This example checks punctuation deletion."
        answer = del_punct.augment(query)

        self.assertEqual(answer, "This example checks punctuation deletion")
    
    def test_delete_sentence(self):
        delete_sentence = DelSent(alpha=1.0)
        para = "This is a test sentence 1. This is test sentence 2."
        ans = delete_sentence.augment(para)

        self.assertEqual("", ans)

    def test_lemmatization(self):
        lemma = Lemmatize()
        query = "This example is testing lemmatization"
        answer = lemma.augment(query)
        
        self.assertNotEqual(answer, query)

    def test_shuffle_words(self):
        shuff_words = ShufWords()
        query = "This is a test sentence"
        answer = shuff_words.augment(query)
        self.assertNotEqual(query, answer)

    def test_shuffle_sents(self):
        shuffle_words_sents = ShufWordsKeepSents(alpha=1.0)
        query = "This is test sentence 1. And this is test sentence 2."
        answer = shuffle_words_sents.augment(query)
        
        self.assertNotEqual(query, answer)

    def test_shuf_words_keep_sents_np(self):
        shuffle_obj = ShufWordsKeepSentsAndNPs(alpha=1.0)
        query = "Apple is red in color. Sky is blue in color."
        answer = shuffle_obj.augment(query)
        
        self.assertNotEqual(query, answer)
        
    def test_shuffle_words_keep_np(self):
        shuffle_obj = ShufWordsKeepNPs()
        query = "This sentence checks the implementation of the function ShufWordsKeepNPs"
        answer = shuffle_obj.augment(query)
        
        self.assertNotEqual(query, answer)

    def test_shuffle_np_slots(self):
        shuffle_obj = ShufNPSlots()
        query = "Example A and Example B are examples of nouns"
        answer = shuffle_obj.augment(query)
        
        self.assertNotEqual(query, answer)

    def test_shuffle_preps(self):
        shuffle_obj = ShufPrepositions()
        query = "Sentence 2 in after Sentence 1 and before Sentence 3"
        answer = shuffle_obj.augment(query)

        self.assertNotEqual(query, answer)


    def test_reverse_np(self):
        reverse_obj = ReverseNPSlots()
        query = "Example A and Example B are examples of nouns"
        answer = reverse_obj.augment(query)

        self.assertNotEqual(query, answer)

    def test_shuffle_sentences(self):
        shuffle_obj = ShufSents()
        query = "This is test sentence 1. This is test sentence 2. This is test sentence 3."
        answer = shuffle_obj.augment(query)
        
        if answer == query:
            answer = shuffle_obj.augment(query)
        
        self.assertNotEqual(answer, query)

    def test_reverse_sentences(self):
        reverse_obj = ReverseSents()
        query = "This is test sentence1. This is test sentence 2."
        answer = reverse_obj.augment(query)

        self.assertEqual(answer, "This is test sentence 2. This is test sentence1.")

    def test_reverse_words(self):
        reverse_obj = ReverseWords()
        query = "this augmentation reverses all the words"
        answer = reverse_obj.augment(query)

        self.assertEqual(answer, "words the all reverses augmentation this")

    def test_remove_stopwords(self):
        remove_obj = RmStops()
        query = "This is sentence 1 and this is sentence 2"
        answer = remove_obj.augment(query)

        self.assertNotEqual(answer, query)

if __name__ == "__main__":
    unittest.main()

    