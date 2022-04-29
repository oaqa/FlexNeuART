import string
import random
import itertools
from base_class import DataAugment
import spacy

'''
convert text to lowercase
Text -> AppLE
Transform -> apple
'''
class CaseFold(DataAugment):
    def augment(self, text):
        return text.lower()

'''
remove punctuations
Text -> RandomCompany's stock fell by 10%.
Transform -> RandomCompanys stock fell by 10
'''
class DelPunct(DataAugment):
    def __init__(self):
        super().__init__()
        self.trans_punct = str.maketrans('', '', string.punctuation)

    def augment(self, text):
        return text.translate(self.trans_punct)

'''
delete a sentence from given/random position
Text -> Lionel Messi was born in Argentina. He currently plays for PSG.
Transform (delete first) -> He currently plays for PSG.
'''
class DelSent(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.position = kwargs.get('position', 'rand')
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
        self.rng = random.Random(self.random_seed)

    def augment(self, text):
        sents = list(self.nlp(text).sents)
        if len(sents) > 1: # don't remove if only 1 sentence
            if self.position == 'start':
                sents = sents[1:]
            elif self.position == 'end':
                sents = sents[:-1]
            elif self.position == 'rand':
                pos = self.rng.randrange(len(sents))
                sents = sents[:pos] + sents[pos+1:]
            else:
                raise ValueError()
            return ' '.join(str(s) for s in sents)
        return None


'''
lemmatize every token except stopwords
Text -> Oreo likes to play in the park every evening.
Transform -> Oreo like to play in the park every evening .
'''
class Lemmatize(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
    
    def augment(self, text):
        dtext_b = [t.lemma_ if not t.is_stop else t for t in self.nlp(text)]
        return ' '.join(str(s) for s in dtext_b)


'''
shuffle words in a sentence
Text -> Camp Nou is the home of football club barcelona
Transform -> the football club home barcelona is of Camp Nou
'''
class ShufWords(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.rng = random.Random(self.random_seed)
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
    
    def augment(self, text):
        dtoks = [str(t) for t in self.nlp(text)]
        self.rng.shuffle(dtoks)
        return ' '.join(str(s) for s in dtoks)


'''
shuffle words within sentences
can take a paragraph as input and shuffle words within sentences
Text ->
Argentina lost to germany in the 2014 world cup final. Mario Gotze scored an extra time goal.
Transform ->
the world lost Argentina in germany cup 2014 to final . scored time Mario Gotze extra goal an .
'''
class ShufWordsKeepSents(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
        self.rng = random.Random(str(self.random_seed))

    def augment(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
        dsents = []
        for sent in self.nlp(text).sents:
            sent_toks = [str(s) for s in sent[:-1]]
            self.rng.shuffle(sent_toks)
            sent_toks = sent_toks + [str(sent[-1])]
            dsents.append(' '.join(sent_toks))
        return ' '.join(dsents)

'''
shuffle sentences but sentence order and order of words in a noun chunk is preserved
text = "Argentina lost to germany in the 2014 world cup final. Mario Gotze scored an extra time goal."
noun_chunks = [Argentina, germany, the 2014 world cup, Mario Gotze, an extra time goal]
noun_chunk_idsxs = {0, 3, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17}

Text ->
Argentina lost to germany in the 2014 world cup final. Mario Gotze scored an extra time goal.
Transform ->
final Argentina in the 2014 world cup lost germany to . Mario Gotze scored an extra time goal .
'''
class ShufWordsKeepSentsAndNPs(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.rng = random.Random(str(self.random_seed))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))

    def augment(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
        dsents = []
        parsed_text = self.nlp(text)
        noun_chunks = list(parsed_text.noun_chunks)
        noun_chunk_idxs = set(itertools.chain(*(range(c.start, c.end) for c in noun_chunks)))
        for sent in parsed_text.sents:
            these_noun_chunks = [str(c) for c in noun_chunks if c.start >= sent.start and c.end <= sent.end]
            these_non_noun_chunks = [str(parsed_text[i]) for i in range(sent.start, sent.end - 1) if i not in noun_chunk_idxs]
            sent_toks = these_noun_chunks + these_non_noun_chunks
            self.rng.shuffle(sent_toks)
            sent_toks = sent_toks + [str(sent[-1])]
            dsents.append(' '.join(sent_toks))
        return ' '.join(dsents)

'''
Similar to ShufWordsKeepSentsAndNPs
Sentence order not maintained
Text ->
Barcelona are the first club in history to win the treble twice. Bayern are the second club to so.
Transform ->
are to Bayern . history so Barcelona the first club in the second club win are the treble twice to .
'''
class ShufWordsKeepNPs(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))

    def augment(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
        parsed_text = self.nlp(text)
        noun_chunks = list(parsed_text.noun_chunks)
        noun_chunk_idxs = set(itertools.chain(*(range(c.start, c.end) for c in noun_chunks)))
        noun_chunks = [str(c) for c in noun_chunks]
        non_noun_chunks = [str(t) for i, t in enumerate(parsed_text) if i not in noun_chunk_idxs]
        toks = noun_chunks + non_noun_chunks
        self.rng.shuffle(toks)
        return ' '.join(toks)


'''
Shuffle noun chunks and preserve the rest of the sentence
Text ->
Lionel Messi and Cristiano Ronaldo are the amonst the best football players of all time.
Transform ->
the best football players are the amonst Lionel Messi and Cristiano Ronaldo of all time .
'''
class ShufNPSlots(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.rng = random.Random(str(self.random_seed))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
    
    def augment(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
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
        self.rng.shuffle(noun_chunks)
        toks = []
        for chunk in chunks:
            if isinstance(chunk, int):
                toks.append(str(noun_chunks[chunk]))
            else:
                toks.append(chunk)
        return ' '.join(toks)

'''
Shuffle the prepositions in the sentences
Text -> The apple is in front of the table beside the ball
Transform -> The apple is of front in the table beside the ball
'''
class ShufPrepositions(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.rng = random.Random(str(self.random_seed))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
    
    def augment(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
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
        self.rng.shuffle(preps)
        toks = []
        for chunk in chunks:
            if isinstance(chunk, int):
                toks.append(str(preps[chunk]))
            else:
                toks.append(chunk)
        return ' '.join(toks)

'''

***************************************
VERIFY THE IMPLEMENTATION AND RATIONALE
***************************************

Sample random positions and shuffle them
Number of positions sampled is equal to number of noun chunks
Text -> Joey and Chandler lost Ben on the bus
Transform -> lost and Chandler bus Ben on the Joey
'''
class SwapNumNPSlots2(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.rng = random.Random(str(self.random_seed))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
    
    def augment(self, text):
        self.rng = random.Random(str(self.random_seed) + text)
        parsed_text = self.nlp(text)
        num_swaps = len(list(parsed_text.noun_chunks))
        toks = [str(t) for t in parsed_text]
        new_toks = [str(t) for t in parsed_text]
        positions = self.rng.sample(range(len(toks)), k=num_swaps)
        shuf_positions = list(positions)
        self.rng.shuffle(shuf_positions)
        for old, new in zip(positions, shuf_positions):
            new_toks[new] = toks[old]
        return ' '.join(new_toks)


'''
Reverse the noun chunks of the text
Text -> Joey and Chandler live next to Monica and Rachael
Transform -> Rachael and Monica live next to Chandler and Joey
'''
class ReverseNPSlots(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.rng = random.Random(str(self.random_seed))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))

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

'''
Shuffle sentences
Text ->
Barcelona play at the Camp Nou. Real play at the Santiago Bernabeu. None of them play at Wembley.
Transform ->
Real play at the Santiago Bernabeu. Barcelona play at the Camp Nou. None of them play at Wembley.
'''
class ShufSents(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.rng = random.Random(str(self.random_seed))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
    
    def augment(self, text):
        dsents = [str(s) for s in self.nlp(text).sents]
        self.rng.shuffle(dsents)
        dtext_b = ' '.join(str(s) for s in dsents)
        return dtext_b

'''
Reverse Sentences
Text -> There are twelve months in a year. Not all of them have 30 days.
Transform -> Not all of them have 30 days. There are twelve months in a year.
'''
class ReverseSents(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.rng = random.Random(str(self.random_seed))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
    
    def augment(self, text):
        dsents = [str(s) for s in self.nlp(text).sents]
        dtext_b = ' '.join(str(s) for s in reversed(dsents))
        return dtext_b

'''
Reverse words of a given chunk of text
Text -> this transformation reverses all the words
Transform -> words the all reverses transformation this
'''
class ReverseWords(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.rng = random.Random(str(self.random_seed))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
    
    def augment(self, text):
        dtext_b = [str(s) for s in reversed(self.nlp(text))]
        return ' '.join(dtext_b)



'''
Remove Stopwords
Text -> The Night King wanted to kill bran but Arya killed him
Transform -> Night King wanted kill bran Arya killed
'''
class RmStops(DataAugment):
    def __init__(self, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.rng = random.Random(str(self.random_seed))
        self.nlp = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
    
    def augment(self, text):
        stopwords = self.nlp.Defaults.stop_words
        terms = [str(t) for t in self.nlp(text) if str(t).lower() not in stopwords]
        return ' '.join(terms)


'''
Function not imported yet

Add text to text (decide what to add)
class AddSent(Transform, SpacyMixin, RandomMixin):
    def __init__(self, position='start', rel=0, **kwargs):
        super().__init__(kwargs.get('random_seed',  42))
        self.position = position
        self.rel = rel

    def augment(self, text):
        raise NotImplementedError


Fix typos in sentece (terrier dependency)
class Typo(Transform, RandomMixin):
    def __init__(self, no_stops=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._typo_list = None
        self._typo_regex = None
        self._no_stops = no_stops

    def typo_list(self):
        if self._typo_list is None:
            self._typo_list = {}
            if self._no_stops:
                J.initialize()
                stopwords = J._autoclass("org.terrier.terms.Stopwords")(None)
            for line in open('etc/wiki_typos.tsv'):
                typo, corrects = line.rstrip().split('\t')
                corrects = corrects.split(', ')
                for correct in corrects:
                    if self._no_stops and stopwords.isStopword(correct.lower()):
                        continue
                    if correct not in self._typo_list:
                        self._typo_list[correct] = []
                    self._typo_list[correct].append(typo)
            self._typo_regex = '|'.join(re.escape(c) for c in self._typo_list)
            self._typo_regex = re.compile(f'\\b({self._typo_regex})\\b')
        return self._typo_list, self._typo_regex

    def augment(self, text):
        typos, regex = self.typo_list()
        match = regex.search(text)
        while match:
            typo_candidates = typos[match.group(1)]
            if len(typo_candidates) > 1:
                typo = self.rng.choice(typo_candidates)
            else:
                typo = typo_candidates[0]
            text = text[:match.start()] + typo + text[match.end():]
            match = regex.search(text)
        return text
'''