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
        query = "RandomCompany's stock fell by 10%."
        answer = del_punct.augment(query)

        self.assertEqual(answer, "RandomCompanys stock fell by 10")
    
    def test_delete_sentence(self):
        del_first = DelSent(position='start')
        del_last = DelSent(position='end')
        para = "Lionel Messi was born in Argentina. He currently plays for PSG."
        ans1 = del_first.augment(para)
        ans2 = del_last.augment(para)

        self.assertEqual(ans2, "Lionel Messi was born in Argentina.")
        self.assertEqual(ans1, "He currently plays for PSG.")

    def test_lemmatization(self):
        lemma = Lemmatize()
        query = "Oreo likes to play in the park every evening."
        answer = lemma.augment(query)
        
        self.assertEqual(answer, "Oreo like to play in the park every evening .")

    def test_shuffle_words(self):
        shuff_words = ShufWords()
        query = "Camp Nou is the home of football club barcelona"
        answer = shuff_words.augment(query)
        self.assertNotEqual(query, answer)

    def test_shuffle_sents(self):
        shuffle_words_sents = ShufWordsKeepSents()
        query = "Argentina lost to germany in the 2014 world cup final. Mario Gotze scored an extra time goal."
        answer = shuffle_words_sents.augment(query)
        word_list = answer.split(" ")

        self.assertTrue(word_list.index("scored") > 9)
        self.assertTrue(word_list.index("lost") <= 9)

    def test_shuf_words_keep_sents_np(self):
        shuffle_obj = ShufWordsKeepSentsAndNPs()
        query = "Argentina lost to germany in the 2014 world cup final. Mario Gotze scored an extra time goal."
        answer = shuffle_obj.augment(query)
        word_list = answer.split(" ")

        self.assertTrue(word_list.index("Mario")+1 == word_list.index("Gotze"))
        self.assertTrue(word_list.index("lost") < word_list.index("scored"))
        
    def test_shuffle_words_keep_np(self):
        shuffle_obj = ShufWordsKeepNPs()
        query = "Barcelona are the first club in history to win the treble twice. Bayern are the second club to so."
        answer = shuffle_obj.augment(query)
        words_list = answer.split(" ")

        self.assertTrue(words_list.index("first")+1 == words_list.index("club"))

    def test_shuffle_np_slots(self):
        shuffle_obj = ShufNPSlots()
        query = "Lionel Messi and Cristiano Ronaldo are the amonst the best football players of all time."
        answer = shuffle_obj.augment(query)
        word_list = answer.split(" ")
        
        self.assertTrue(word_list.index("Messi")-1 == word_list.index("Lionel"))
        self.assertNotEqual(answer, query)

    def test_shuffle_preps(self):
        shuffle_obj = ShufPrepositions()
        query = "The apple is in front of the table beside the ball"
        answer = shuffle_obj.augment(query)

        self.assertNotEqual(query, answer)

    def test_swap_np_slots(self):
        swap_obj = SwapNumNPSlots2()
        query = "Joey and Chandler lost Ben on the bus"
        answer = swap_obj.augment(query)

        self.assertNotEqual(answer, query)

    def test_reverse_np(self):
        reverse_obj = ReverseNPSlots()
        query = "Joey and Chandler live next to Monica and Rachael"
        answer = reverse_obj.augment(query)

        self.assertEqual(answer, "Rachael and Monica live next to Chandler and Joey")

    def test_shuffle_sentences(self):
        shuffle_obj = ShufSents()
        query = "Barcelona play at the Camp Nou. Real play at the Santiago Bernabeu. None of them play at Wembley."
        answer = shuffle_obj.augment(query)

        self.assertNotEqual(answer, query)

    def test_reverse_sentences(self):
        reverse_obj = ReverseSents()
        query = "There are twelve months in a year. Not all of them have 30 days."
        answer = reverse_obj.augment(query)

        self.assertEqual(answer, "Not all of them have 30 days. There are twelve months in a year.")

    def test_reverse_words(self):
        reverse_obj = ReverseWords()
        query = "this augmentation reverses all the words"
        answer = reverse_obj.augment(query)

        self.assertEqual(answer, "words the all reverses augmentation this")

    def test_remove_stopwords(self):
        remove_obj = RmStops()
        query = "The Night King wanted to kill bran but Arya killed him"
        answer = remove_obj.augment(query)

        self.assertEqual(answer, "Night King wanted kill bran Arya killed")

if __name__ == "__main__":
    unittest.main()

    