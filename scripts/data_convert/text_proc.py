import spacy
import re

SPACY_POS = 'tagger'
SPACY_NER = 'ner'
SPACY_PARSER = 'parser'

ALPHANUM_TOKENS = re.compile("^[a-zA-Z-_.0-9]+$")


def isAlphaNum(s):
    return s and (ALPHANUM_TOKENS.match(s) is not None)


"""A wrapper class to handle basic Stanza-based text processing.
   Stanza is a new version of the Stanford Core NLP package"""


class StanzaTextParser:
    def __init__(self, lang, stopWords,
                 removePunct=True,
                 keepOnlyAlphaNum=False,
                 lowerCase=True):
        """Constructor.

        :param  lang         the name of the language
        :param  stopWords    a list of stop words to be excluded (case insensitive);
                             a token is also excluded when its lemma is in the stop word list.
        :param  removePunct  a bool flag indicating if the punctuation tokens need to be removed
        :param  keepOnlyAlphaNum a bool flag indicating if we need to keep only alpha-numeric characters
        """
        import stanza

        self._nlp = stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma')

        self._removePunct = removePunct
        self._stopWords = frozenset([w.lower() for w in stopWords])
        self._keepOnlyAlphaNum = keepOnlyAlphaNum
        self._lowerCase = lowerCase

    @staticmethod
    def _basic_clean(text):
        return text.replace("’", "'")

    def __call__(self, text):
        """A thin wrapper that merely calls stanza.

        :param text     input text string
        :return         a spacy Doc object
        """

        return self._nlp(SpacyTextParser._basic_clean(text))

    def procText(self, text):
        """Process text, remove stopwords and obtain lemmas, but does not split into sentences.
        This function should not emit newlines!

        :param text     input text string
        :return         a tuple (lemmatized text, original-form text). Text is white-space separated.
        """

        lemmas = []
        tokens = []
        doc = self(text)
        for sent in doc.sentences:
            for tokObj in sent.words:
                if self._removePunct and tokObj.upos == 'PUNCT':
                    continue
                lemma = tokObj.lemma
                text = tokObj.text
                if self._keepOnlyAlphaNum and not isAlphaNum(text):
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


"""A wrapper class to handle basic Spacy-based text processing."""


class SpacyTextParser:
    def __init__(self, modelName, stopWords,
                 removePunct=True,
                 sentSplit=False,
                 keepOnlyAlphaNum=False,
                 lowerCase=True,
                 enablePOS=True):
        """Constructor.

        :param  modelName    a name of the spacy model to use, e.g., en_core_web_sm
        :param  stopWords    a list of stop words to be excluded (case insensitive);
                             a token is also excluded when its lemma is in the stop word list.
        :param  removePunct  a bool flag indicating if the punctuation tokens need to be removed
        :param  sentSplit    a bool flag indicating if sentence splitting is necessary
        :param  keepOnlyAlphaNum a bool flag indicating if we need to keep only alpha-numeric characters
        :param  enablePOS    a bool flag that enables POS tagging (which, e.g., can improve lemmatization)
        """

        # Disabling all heavy-weight parsing, but enabling splitting into sentences
        disableList = [SPACY_NER, SPACY_PARSER]
        if not enablePOS:
            disableList.append(SPACY_POS)
        print('Disabled Spacy components: ', disableList)

        self._nlp = spacy.load(modelName, disable=disableList)
        if sentSplit:
            sentencizer = self._nlp.create_pipe("sentencizer")
            self._nlp.add_pipe(sentencizer)

        self._removePunct = removePunct
        self._stopWords = frozenset([w.lower() for w in stopWords])
        self._keepOnlyAlphaNum = keepOnlyAlphaNum
        self._lowerCase = lowerCase

    @staticmethod
    def _basic_clean(text):
        return text.replace("’", "'")

    def __call__(self, text):
        """A thin wrapper that merely calls spacy.

        :param text     input text string
        :return         a spacy Doc object
        """

        return self._nlp(SpacyTextParser._basic_clean(text))

    def procText(self, text):
        """Process text, remove stopwords and obtain lemmas, but does not split into sentences.
        This function should not emit newlines!

        :param text     input text string
        :return         a tuple (lemmatized text, original-form text). Text is white-space separated.
        """

        lemmas = []
        tokens = []
        doc = self(text)
        for tokObj in doc:
            if self._removePunct and tokObj.is_punct:
                continue
            lemma = tokObj.lemma_
            text = tokObj.text
            if self._keepOnlyAlphaNum and not isAlphaNum(text):
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
    def __init__(self, lowerCase):
        self._lowerCase = lowerCase

    def __call_(self, text):
        if self._lowerCase:
            text = text.lower()
        return text.split()


class SpacyTokenExtractor(TokenExtractor):
    def __init__(self, modelName, stopwords, lemmatize,
                 lowerCase=True, keepOnlyAlphaNum=True):
        self._lemmatize = lemmatize

        self._nlp = SpacyTextParser(modelName,
                                    stopwords,
                                    keepOnlyAlphaNum=keepOnlyAlphaNum, lowerCase=lowerCase)

    def __call__(self, text):
        lemmas, unlemm = self._nlp.procText(text)

        return lemmas if self._lemmatize else unlemm


class StanzaTokenExtractor(TokenExtractor):
    def __init__(self, langName, stopwords, lemmatize,
                 lowerCase=True, keepOnlyAlphaNum=True):
        self._lemmatize = lemmatize

        self._nlp = StanzaTextParser(langName,
                                     stopwords,
                                     keepOnlyAlphaNum=keepOnlyAlphaNum, lowerCase=lowerCase)

    def __call__(self, text):
        lemmas, unlemm = self._nlp.procText(text)

        return lemmas if self._lemmatize else unlemm


TOKEN_EXTR_TYPES = {'WhiteSpaceTokenExtractor': WhiteSpaceTokenExtractor,
                    'SpacyTokenExtractor': SpacyTokenExtractor,
                    'StanzaTokenExtractor': StanzaTokenExtractor}


class TokenExtrFactory:
    @staticmethod
    def create(tokExtrType, **kwargs):
        if not tokExtrType in TOKEN_EXTR_TYPES:
            raise Exception('Unrecognized token extractor type: ' + tokExtrType)
        return TOKEN_EXTR_TYPES[tokExtrType](**kwargs)
