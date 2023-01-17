conf = {
    "random_character_insertion": {
        "word_add_probability": 0.1,
        "character_add_probability": 0.1
    },
    "random_character_deletion": {
        "word_remove_probability": 0.1,
        "character_remove_probability": 0.1
    },
    "random_character_swap": {
        "word_swap_probability": 0.1,
        "character_swap_probability": 0.1
    },
    "random_character_replace": {
        "word_replace_probability": 0.1,
        "character_replace_probability": 0.1
    },
    "keyboard_character_insertion": {
        "word_add_probability": 0.1,
        "character_add_probability": 0.1
    },
    "keyboard_character_replace": {
        "word_replace_probability": 0.1,
        "character_replace_probability": 0.1
    },
    "random_word_deletion": {
        "probability": 0.1
    },
    "random_word_insertion": {
        "probability": 0.1
    },
    "random_word_swap": {
        "probability": 0.1
    },
    "document_constant_length": {
        "doc_length": 1000
    },
    "document_cut_out": {
        "p": 0.1,
        "span_p": 0.2
    },
    "query_cut_out": {
        "p": 0.5
    },
    "synonym_word_replacement": {
        "probability": 0.05
    },
    "hypernym_word_replacement": {
        "probability": 0.05
    },
    "query_cache": {
        "cache_path": "/home/ubuntu/efs/capstone/data_aug/cache_data/queries_100.json"
    },
    "del_sentence": {
        "spacy_model": "en_core_web_sm",
        "alpha": 0.1
    },
    "lemmatize": {
        "spacy_model": "en_core_web_sm",
        "alphs": 0.1
    },
    "shuffle_words": {
        "spacy_model": "en_core_web_sm"
    },
    "shuffle_words_keep_sentences": {
        "spacy_model": "en_core_web_sm",
        "alpha": 0.1
    },
    "shuffle_words_keep_sent_and_nps": {
        "spacy_model": "en_core_web_sm",
        "alpha": 0.1
    },
    "shuf_words_keep_noun_phrase": {
        "spacy_model": "en_core_web_sm"
    },
    "shuf_noun_phrase": {
        "spacy_model": "en_core_web_sm"
    },
    "shuf_prepositions": {
        "spacy_model": "en_core_web_sm"
    },
    "reverse_noun_phrase_slots": {
        "spacy_model": "en_core_web_sm"
    },
    "shuffle_sentences": {
        "spacy_model": "en_core_web_sm"
    },
    "reverse_sentences": {
        "spacy_model": "en_core_web_sm"
    },
    "reverse_words": {
        "spacy_model": "en_core_web_sm"
    },
    "remove_stopwords": {
        "spacy_model": "en_core_web_sm"
    }
}