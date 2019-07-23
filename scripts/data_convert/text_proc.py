import spacy
import re

SPACY_POS = 'tagger'
SPACY_NER = 'ner'
SPACY_PARSER = 'parser'

ALPHANUM_TOKENS = re.compile("^[a-zA-Z-_.0-9]+$")

def isAlphaNum(s):
  return s and (ALPHANUM_TOKENS.match(s) is not None)

"""A wrapper class to handle basic Spacy-based text processing."""
class SpacyTextParser:
  def __init__(self, spacyModel, stopWords, removePunct=True, sentSplit=False, keepOnlyAlphaNum=False):
    """Constructor.

    :param  spacyMode    a name of the spacy model to use, e.g., en_core_web_sm 
    :param  stopWords    a list of stop words to be excluded (case insensitive); 
                         a token is also excluded when its lemma is in the stop word list.
    :param  removePunct  a bool flag indicating if the punctuation tokens need to be removed
    :param  sentSplit    a bool flag indicating if sentence splitting is necessary
    :param  keepOnlyAlphaNum a bool flag indicating if we need to keep only alpha-numeric characters
    """

    # Disabling all heavy-weight parsing, but enabling splitting into sentences
    self._nlp = spacy.load(spacyModel, disable=[SPACY_NER, SPACY_PARSER, SPACY_POS])
    if sentSplit:
      sentencizer = self._nlp.create_pipe("sentencizer")
      self._nlp.add_pipe(sentencizer)

    self._removePunct = removePunct
    self._stopWords = frozenset([w.lower() for w in stopWords])
    self._keepOnlyAlphaNum = keepOnlyAlphaNum


  def __call__(self, text): 
    """A thin wrapper that merely calls spacy.

    :param text     input text string
    :return         a spacy Doc object 
    """

    return self._nlp(text)

  
  def procText(self, text):
    """Process text, remove stopwords and obtain lemmas, but does not split into sentences.
    :param text     input text string
    :return         a tuple (lemmatized text, original-form text). Text is white-space separated.
    """
    text = text.replace("’", "'")

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

      lemmas.append(lemma)
      tokens.append(text)

    return ' '.join(lemmas), ' '.join(tokens)
