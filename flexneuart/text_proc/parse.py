#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import spacy
import re
import krovetzstemmer
import urllib

from flexneuart.config import BERT_BASE_MODEL

"""
    Various procedureds/classes to parse text (including tokenization)
"""

SPACY_POS = 'tagger'
SPACY_NER = 'ner'
SPACY_PARSER = 'parser'

ALPHANUM_TOKENS = re.compile("^[a-zA-Z-_.0-9]+$")


def is_alpha_num(s):
    return s and (ALPHANUM_TOKENS.match(s) is not None)


class Sentencizer:
    """A simple wrapper for the rule-based Spacy sentence splitter."""
    def __init__(self, model_name):
        """
        :param model_name: a name of the spacy model to use, e.g., en_core_web_sm
        """
        self._nlp = spacy.load(model_name, disable=[SPACY_NER, SPACY_PARSER, SPACY_POS])
        self._nlp.add_pipe("sentencizer")

    def __call__(self, text):
        """A thin wrapper that merely calls spacy.

            :param text     input text string
            :return         a spacy Doc object
        """
        return self._nlp(text).sents


class KrovetzStemParser:
    """
        A simple tokenizer that truncates words using a Krovetz stemmer.
        It works well only for a pretty clean text.
        It lower-cases text (seems to be necessary for the stemmer).
    """
    def __init__(self, stop_words):
        """Constructor.

        :param stop_words: a list of stop words to be excluded (case insensitive),
                           which is applied *BEFORE* stemming.
        :param lower_case: lower-case input if True
        """
        self.regex_drop_char = re.compile('[^a-z0-9\s]+')
        self.regex_multi_space = re.compile('\s+')
        self.stemmer = krovetzstemmer.Stemmer()
        self.stop_words = frozenset([w.lower() for w in stop_words])

    def __call__(self, text):
        text = text.lower()
        s = self.regex_multi_space.sub(' ', self.regex_drop_char.sub(' ', text)).strip()
        s = ' '.join([self.stemmer(t) for t in s.split() if t not in self.stop_words])
        return s


class SpacyTextParser:
    """
        A wrapper class to handle basic Spacy-based text processing.
        It also implements sentence splitting, but there's a design flaw,
        which makes it hard to obtain both sentences and other data.
        So, instead please use the separate sentencizer class instead.
    """
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
        :param  lower_case    lower-case input if True
        :param  enable_pos    a bool flag that enables POS tagging (which, e.g., can improve lemmatization)
        """

        # Disabling all heavy-weight parsing, but enabling splitting into sentences
        disable_list = [SPACY_NER, SPACY_PARSER]
        if not enable_pos:
            disable_list.append(SPACY_POS)
        print('Disabled Spacy components: ', disable_list)

        self._nlp = spacy.load(model_name, disable=disable_list)
        if sent_split:
            self._nlp.add_pipe("sentencizer")

        self._removePunct = remove_punct
        self._stopWords = frozenset([w.lower() for w in stop_words])
        self._keepOnlyAlphaNum = keep_only_alpha_num
        self._lowerCase = lower_case

    @staticmethod
    def _basic_clean(text):
        # TODO see a comment to the function __call__
        return text.replace("’", "'")

    def __call__(self, text):
        """A thin wrapper that merely calls spacy.

        :param text     input text string
        :return         a spacy Doc object
        """
        # TODO / important note. Currently, this function
        # does not change offsets, but we should probably overall
        # avoid using it and support some generic basic cleanup
        # function in a different way
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


def pretokenize_url(url):
    """A hacky procedure to "pretokenize" URLs.

    :param  url:  an input URL
    :return a URL with prefixes (see below) removed and some characters replaced with ' '
    """
    remove_pref = ['http://', 'https://', 'www.']
    url = urllib.parse.unquote(url)
    changed = True
    while changed:
        changed = False
        for p in remove_pref:
            assert len(p) > 0
            if url.startswith(p):
                changed = True
                url = url[len(p):]
                break

    return re.sub(r'[.,:!\?/"+\-\'=_{}()|]', " ", url)


def get_bert_tokenizer():
    """
    Returns: a fast BERT tokenizer
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BERT_BASE_MODEL)


def get_retokenized(tokenizer, text):
    """Obtain a space separated re-tokenized text.
    :param tokenizer:  a tokenizer that has the function
                       tokenize that returns an array of tokens.
    :param text:       a text to re-tokenize.
    """
    return ' '.join(tokenizer.tokenize(text))


def add_retokenized_field(data_entry,
                        src_field,
                        dst_field,
                        tokenizer):
    """
    Create a re-tokenized field from an existing one.

    :param data_entry:   a dictionary of entries (keys are field names, values are text items)
    :param src_field:    a source field
    :param dst_field:    a target field
    :param tokenizer:    a tokenizer to use, if None, nothing is done
    """
    if tokenizer is not None:
        dst = ''
        if src_field in data_entry:
            dst = get_retokenized(tokenizer, data_entry[src_field])

        data_entry[dst_field] = dst






