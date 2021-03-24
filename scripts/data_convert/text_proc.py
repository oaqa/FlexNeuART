import spacy
import re

SPACY_POS = 'tagger'
SPACY_NER = 'ner'
SPACY_PARSER = 'parser'

ALPHANUM_TOKENS = re.compile("^[a-zA-Z-_.0-9]+$")

def is_alpha_num(s):
    return s and (ALPHANUM_TOKENS.match(s) is not None)


"""
    A wrapper class to handle basic Stanza-based text processing.
    Stanza is a new version of the Stanford Core NLP package.
    However, it seems to be a tad too slow so Stanza installation is not 
    required and StanzaTextParser is not imported by default.
"""
class StanzaTextParser:
    def __init__(self, lang, stop_words,
                 remove_punct=True,
                 keep_only_alpha_num=False,
                 lower_case=True):
        """Constructor.

        :param  lang         the name of the language
        :param  stop_words    a list of stop words to be excluded (case insensitive);
                             a token is also excluded when its lemma is in the stop word list.
        :param  remove_punct  a bool flag indicating if the punctuation tokens need to be removed
        :param  keep_only_alpha_num a bool flag indicating if we need to keep only alpha-numeric characters
        """
        import stanza

        self._nlp = stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma')

        self._removePunct = remove_punct
        self._stopWords = frozenset([w.lower() for w in stop_words])
        self._keepOnlyAlphaNum = keep_only_alpha_num
        self._lowerCase = lower_case

    @staticmethod
    def _basic_clean(text):
        return text.replace("’", "'")

    def __call__(self, text):
        """A thin wrapper that merely calls stanza.

        :param text     input text string
        :return         a spacy Doc object
        """

        return self._nlp(SpacyTextParser._basic_clean(text))

    def proc_text(self, text):
        """Process text, remove stopwords and obtain lemmas, but does not split into sentences.
        This function should not emit newlines!

        :param text     input text string
        :return         a tuple (lemmatized text, original-form text). Text is white-space separated.
        """

        lemmas = []
        tokens = []
        doc = self(text)
        for sent in doc.sentences:
            for tok_obj in sent.words:
                if self._removePunct and tok_obj.upos == 'PUNCT':
                    continue
                lemma = tok_obj.lemma
                text = tok_obj.text
                if self._keepOnlyAlphaNum and not is_alpha_num(text):
                    continue
                tok1 = text.lower()
                tok2 = lemma.lower()
                if tok1 in self._stopWords or tok2 in self._stopWords:
                    continue

                if self._lowerCase:
                    text = text.lower()
                    lemma = lemma.lower()

                lemmas.append(lemma)
                tokens.append(text)

        return ' '.join(lemmas), ' '.join(tokens)


class Sentencizer:
    """A simple wrapper for the rule-based Spacy sentence splitter."""
    def __init__(self, model_name):
        """
        :param model_name: a name of the spacy model to use, e.g., en_core_web_sm
        """
        self._nlp = spacy.load(model_name, disable=[SPACY_NER, SPACY_PARSER,SPACY_POS])
        self._nlp.add_pipe(self._nlp.create_pipe("sentencizer"))

    def __call__(self, text):
        """A thin wrapper that merely calls spacy.

            :param text     input text string
            :return         a spacy Doc object
        """
        return self._nlp(text).sents


"""A wrapper class to handle basic Spacy-based text processing.
   It also implements sentence splitting, but there's a design flaw,
   which makes it hard to obtain both sentences and other data.
   So, instead please use the separate sentencizer class instead.
"""
class SpacyTextParser:
    def __init__(self, model_name, stop_words,
                 remove_punct=True,
                 sent_split=False,
                 keep_only_alpha_num=False,
                 lower_case=True,
                 enable_pos=True):
        """Constructor.

        :param  model_name    a name of the spacy model to use, e.g., en_core_web_sm
        :param  stop_words    a list of stop words to be excluded (case insensitive);
                             a token is also excluded when its lemma is in the stop word list.
        :param  remove_punct  a bool flag indicating if the punctuation tokens need to be removed
        :param  sent_split    a bool flag indicating if sentence splitting is necessary
        :param  keep_only_alpha_num a bool flag indicating if we need to keep only alpha-numeric characters
        :param  enable_pos    a bool flag that enables POS tagging (which, e.g., can improve lemmatization)
        """

        # Disabling all heavy-weight parsing, but enabling splitting into sentences
        disable_list = [SPACY_NER, SPACY_PARSER]
        if not enable_pos:
            disable_list.append(SPACY_POS)
        print('Disabled Spacy components: ', disable_list)

        self._nlp = spacy.load(model_name, disable=disable_list)
        if sent_split:
            sentencizer = self._nlp.create_pipe("sentencizer")
            self._nlp.add_pipe(sentencizer)

        self._removePunct = remove_punct
        self._stopWords = frozenset([w.lower() for w in stop_words])
        self._keepOnlyAlphaNum = keep_only_alpha_num
        self._lowerCase = lower_case

    @staticmethod
    def _basic_clean(text):
        return text.replace("’", "'")

    def __call__(self, text):
        """A thin wrapper that merely calls spacy.

        :param text     input text string
        :return         a spacy Doc object
        """

        return self._nlp(SpacyTextParser._basic_clean(text))

    def proc_text(self, text):
        """Process text, remove stopwords and obtain lemmas, but does not split into sentences.
        This function should not emit newlines!

        :param text     input text string
        :return         a tuple (lemmatized text, original-form text). Text is white-space separated.
        """

        lemmas = []
        tokens = []
        doc = self(text)
        for tok_obj in doc:
            if self._removePunct and tok_obj.is_punct:
                continue
            lemma = tok_obj.lemma_
            text = tok_obj.text
            if self._keepOnlyAlphaNum and not is_alpha_num(text):
                continue
            tok1 = text.lower()
            tok2 = lemma.lower()
            if tok1 in self._stopWords or tok2 in self._stopWords:
                continue

            if self._lowerCase:
                text = text.lower()
                lemma = lemma.lower()

            lemmas.append(lemma)
            tokens.append(text)

        return ' '.join(lemmas), ' '.join(tokens)


"""A parent class of all token extractors.
Token extractors can be merely white-space tokenizers,
but they can also be more sophisticated processors."""


class TokenExtractor:
    def __init__(self):
        pass

    def __call_(self, text):
        """Process input text, return a list of extracted tokens.

        :param text:  input text.
        :return:  a list of tokens.
        """
        raise NotImplemented


# A possible TODO:
# these token extractors aren't used yet
class WhiteSpaceTokenExtractor(TokenExtractor):
    def __init__(self, lower_case):
        self._lowerCase = lower_case

    def __call_(self, text):
        if self._lowerCase:
            text = text.lower()
        return text.split()


class SpacyTokenExtractor(TokenExtractor):
    def __init__(self, model_name, stopwords, lemmatize,
                 lower_case=True, keep_only_alpha_num=True):
        self._lemmatize = lemmatize

        self._nlp = SpacyTextParser(model_name,
                                    stopwords,
                                    keep_only_alpha_num=keep_only_alpha_num, lower_case=lower_case)

    def __call__(self, text):
        lemmas, unlemm = self._nlp.proc_text(text)

        return lemmas if self._lemmatize else unlemm


class StanzaTokenExtractor(TokenExtractor):
    def __init__(self, lang_name, stopwords, lemmatize,
                 lower_case=True, keep_only_alpha_num=True):
        self._lemmatize = lemmatize

        self._nlp = StanzaTextParser(lang_name,
                                     stopwords,
                                     keep_only_alpha_num=keep_only_alpha_num, lower_case=lower_case)

    def __call__(self, text):
        lemmas, unlemm = self._nlp.proc_text(text)

        return lemmas if self._lemmatize else unlemm


TOKEN_EXTR_TYPES = {'WhiteSpaceTokenExtractor': WhiteSpaceTokenExtractor,
                    'SpacyTokenExtractor': SpacyTokenExtractor,
                    'StanzaTokenExtractor': StanzaTokenExtractor}


class TokenExtrFactory:
    @staticmethod
    def create(tok_extr_type, **kwargs):
        if not tok_extr_type in TOKEN_EXTR_TYPES:
            raise Exception('Unrecognized token extractor type: ' + tok_extr_type)
        return TOKEN_EXTR_TYPES[tok_extr_type](**kwargs)
