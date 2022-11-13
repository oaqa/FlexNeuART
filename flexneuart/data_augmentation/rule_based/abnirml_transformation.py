from flexneuart.data_augmentation.rule_based.data_augment import DataAugment
import string
import random
import itertools
import spacy
from flexneuart.data_augmentation import register_augmentation
import json

"""
The augmentation techniques implemented and code in this file is inspired by -

Sean MacAvaney, Sergey Feldman, Nazli Goharian, Doug Downey, Arman Cohan;
ABNIRML: Analyzing the Behavior of Neural IR Models.
Transactions of the Association for Computational Linguistics 2022; 10 224–239.
doi: https://doi.org/10.1162/tacl_a_00457
"""

@register_augmentation("case_fold")
class CaseFold(DataAugment):
    """
    A class to convert the document to lower case
    ...

    Attributes
    ----------
    None

    Methods
    -------
    augment(text)
        returns the lowercased text
    """
    def __init__(self, name, conf):
        super().__init__(name)

    def augment(self, text):
        return text.lower()

@register_augmentation("del_punctuation")
class DelPunct(DataAugment):
    """
    A class used for removing all punctuations from the document
    ...

    Attributes
    ----------
    trans_punct : dict
        A dict that maps the ASCII values of string.punctuations to None

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        self.trans_punct = str.maketrans('', '', string.punctuation)

    def augment(self, text):
        return text.translate(self.trans_punct)

    
@register_augmentation("del_sentence")
class DelSent(DataAugment):
    """
    A class that that would delete a sentence in the document with a given probability
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang
    alpha : float
        The probability of deleting a sentence

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
            self.alpha = conf[self.augmentation_name]["alpha"]
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm",
                                "alpha": 0.1}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))

    def augment(self, text):
        sents = list(self.nlp(text).sents)
        remaining = []
        for sentence in sents:
            rn = random.uniform(0, 1)
            if rn > self.alpha:
                remaining.append(sentence.text)
        return " ".join(remaining)


@register_augmentation("lemmatize")
class Lemmatize(DataAugment):
    """
    A class to lemmatize the words in a document
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))

    
    def augment(self, text):
        dtext_b = [t.lemma_ if not t.is_stop else t for t in self.nlp(text)]
        return ' '.join(str(s) for s in dtext_b)


@register_augmentation("shuffle_words")
class ShufWords(DataAugment):
    """
    A class to shuffle the tokens in a sentence
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))
    
    def augment(self, text):
        dtoks = [str(t) for t in self.nlp(text)]
        random.shuffle(dtoks)
        return ' '.join(str(s) for s in dtoks)


@register_augmentation("shuffle_words_keep_sentences")
class ShufWordsKeepSents(DataAugment):
    """
    A class to shuffle the tokens in a sentence with some given probability
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang
    alpha : float
        The probability of shuffling the tokens in a sentence

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
            self.alpha = conf[self.augmentation_name]["alpha"]
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm",
                                "alpha": 0.1}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))


    def augment(self, text):
        dsents = []
        for sent in self.nlp(text).sents:
            rn = random.uniform(0, 1)
            if rn < self.alpha:
                sent_toks = [str(s) for s in sent[:-1]]
                random.shuffle(sent_toks)
                sent_toks = sent_toks + [str(sent[-1])]
                dsents.append(' '.join(sent_toks))
            else:
                dsents.append(sent.text)
        return ' '.join(dsents)


@register_augmentation("shuffle_words_keep_sent_and_nps")
class ShufWordsKeepSentsAndNPs(DataAugment):
    """
    A class to shuffle sentences but sentence order and order of words in a noun chunk is preserved
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang
    alpha : float
        The probability of shuffling a sentence in the document

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
            self.alpha = conf[self.augmentation_name]["alpha"]
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm",
                                "alpha": 0.1}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))
        

    def augment(self, text):
        dsents = []
        parsed_text = self.nlp(text)
        noun_chunks = list(parsed_text.noun_chunks)
        noun_chunk_idxs = set(itertools.chain(*(range(c.start, c.end) for c in noun_chunks)))
        for sent in parsed_text.sents:
            rn = random.uniform(0, 1)
            if rn <= self.alpha:
                these_noun_chunks = [str(c) for c in noun_chunks if c.start >= sent.start and c.end <= sent.end]
                these_non_noun_chunks = [str(parsed_text[i]) for i in range(sent.start, sent.end - 1) if i not in noun_chunk_idxs]
                sent_toks = these_noun_chunks + these_non_noun_chunks
                random.shuffle(sent_toks)
                sent_toks = sent_toks + [str(sent[-1])]
                dsents.append(' '.join(sent_toks))
            else:
                dsents.append(sent.text)
        return ' '.join(dsents)


@register_augmentation("shuf_words_keep_noun_phrase")
class ShufWordsKeepNPs(DataAugment):
    """
    A class to shuffle sentences but but preserve the order of words in a noun chunk
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))

    def augment(self, text):
        parsed_text = self.nlp(text)
        noun_chunks = list(parsed_text.noun_chunks)
        noun_chunk_idxs = set(itertools.chain(*(range(c.start, c.end) for c in noun_chunks)))
        noun_chunks = [str(c) for c in noun_chunks]
        non_noun_chunks = [str(t) for i, t in enumerate(parsed_text) if i not in noun_chunk_idxs]
        toks = noun_chunks + non_noun_chunks
        random.shuffle(toks)
        return ' '.join(toks)


@register_augmentation("shuf_noun_phrase")
class ShufNPSlots(DataAugment):
    """
    A class to shuffle noun chunks and preserve the rest of the sentence
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))
    
    def augment(self, text):
        parsed_text = self.nlp(text)
        noun_chunks = list(parsed_text.noun_chunks)
        noun_chunk_idxs = {}
        for i, np in enumerate(noun_chunks):
            for j in range(np.start, np.end):
                noun_chunk_idxs[j] = i
        chunks = []
        i = 0
        while i < len(parsed_text):
            if i in noun_chunk_idxs:
                chunks.append(noun_chunk_idxs[i])
                i = noun_chunks[noun_chunk_idxs[i]].end
            else:
                chunks.append(str(parsed_text[i]))
                i += 1
        random.shuffle(noun_chunks)
        toks = []
        for chunk in chunks:
            if isinstance(chunk, int):
                toks.append(str(noun_chunks[chunk]))
            else:
                toks.append(chunk)
        return ' '.join(toks)

@register_augmentation("shuf_prepositions")
class ShufPrepositions(DataAugment):
    """
    A class to shuffle the prepositions in the sentences of a document
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))
    
    def augment(self, text):
        parsed_text = self.nlp(text)
        preps = list(t for t in parsed_text if t.pos_ == 'ADP')
        list_text = list(parsed_text)
        prep_idxs = {}
        for i, prep in enumerate(preps):
            prep_idxs[list_text.index(preps[i])] = i
        chunks = []
        i = 0
        while i < len(parsed_text):
            if i in prep_idxs:
                chunks.append(prep_idxs[i])
            else:
                chunks.append(str(parsed_text[i]))
            i += 1
        random.shuffle(preps)
        toks = []
        for chunk in chunks:
            if isinstance(chunk, int):
                toks.append(str(preps[chunk]))
            else:
                toks.append(chunk)
        return ' '.join(toks)

@register_augmentation("reverse_noun_phrase_slots")
class ReverseNPSlots(DataAugment):
    """
    A class to reverse noun chunks in a sentence
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))

    def augment(self, text):
        parsed_text = self.nlp(text)
        noun_chunks = list(parsed_text.noun_chunks)
        noun_chunk_idxs = {}
        for i, np in enumerate(noun_chunks):
            for j in range(np.start, np.end):
                noun_chunk_idxs[j] = i
        chunks = []
        i = 0
        while i < len(parsed_text):
            if i in noun_chunk_idxs:
                chunks.append(noun_chunk_idxs[i])
                i = noun_chunks[noun_chunk_idxs[i]].end
            else:
                chunks.append(str(parsed_text[i]))
                i += 1
        noun_chunks = list(reversed(noun_chunks))
        toks = []
        for chunk in chunks:
            if isinstance(chunk, int):
                toks.append(str(noun_chunks[chunk]))
            else:
                toks.append(chunk)
        return ' '.join(toks)

@register_augmentation("shuffle_sentences")
class ShufSents(DataAugment):
    """
    A class to shuffle the sentences in a document
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))
    
    def augment(self, text):
        dsents = [str(s) for s in self.nlp(text).sents]
        random.shuffle(dsents)
        dtext_b = ' '.join(str(s) for s in dsents)
        return dtext_b

@register_augmentation("register_sentences")
class ReverseSents(DataAugment):
    """
    A class to reverse the sentence order in a document
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))
    
    def augment(self, text):
        dsents = [str(s) for s in self.nlp(text).sents]
        dtext_b = ' '.join(str(s) for s in reversed(dsents))
        return dtext_b

@register_augmentation("reverse_words")
class ReverseWords(DataAugment):
    """
    A class to reverse the word order in the document
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))
    
    def augment(self, text):
        dtext_b = [str(s) for s in reversed(self.nlp(text))]
        return ' '.join(dtext_b)

@register_augmentation("remove_stopwords")
class RmStops(DataAugment):
    """
    A class to remove stopwords from the document
    ...

    Attributes
    ----------
    nlp : Spacy Object
        An object of the class spacy.lang

    Methods
    -------
    augment(text)
        returns the augmented text
    """
    def __init__(self, name, conf):
        super().__init__(name)
        try:
            self.nlp = spacy.load(conf[self.augmentation_name]["spacy_model"])
        except:
            expected_config = {self.augmentation_name :
                               {"spacy_model":  "en_core_web_sm"}}
            raise Exception("ERROR: Config file is missing parameters. Please ensure the following parameters exists - \n%s"
                            % json.dumps(expected_config, indent = 3))
    
    def augment(self, text):
        stopwords = self.nlp.Defaults.stop_words
        terms = [str(t) for t in self.nlp(text) if str(t).lower() not in stopwords]
        return ' '.join(terms)
