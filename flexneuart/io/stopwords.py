from flexneuart.config import STOPWORD_FILE


def read_stop_words(file_name=STOPWORD_FILE, lower_case=True):
    """Reads a list of stopwords from a file. By default the words
       are read from a standard repo location and are lowercased.

      :param file_name a stopword file name
      :param lower_case  a boolean flag indicating if lowercasing is needed.

      :return a list of stopwords
    """
    stop_words = []
    with open(file_name) as f:
        for w in f:
            w = w.strip()
            if w:
                if lower_case:
                    w = w.lower()
                stop_words.append(w)

    return stop_words

