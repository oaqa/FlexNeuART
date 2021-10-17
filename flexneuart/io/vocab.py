import pickle
from collections import Counter

class VocabBuilder:
    """Compile a vocabulary together with token stat. from *WHITE-SPACE* tokenized text."""
    def __init__(self):
        self.total_counter = Counter()
        self.doc_counter = Counter()
        self.doc_qty = 0
        self.tot_qty = 0

    def proc_doc(self, text):
        """White-space tokenize the document, update counters."""
        toks = text.strip().split()
        self.total_counter.update(toks)
        self.doc_counter.update(list(set(toks)))
        self.tot_qty += len(toks)
        self.doc_qty += 1

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            dt = [self.total_counter, self.doc_counter, self.doc_qty, self.tot_qty]
            pickle.dump(dt, f)

    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as f:
            dt = pickle.load(f)
        res = VocabBuilder()
        res.total_counter, res.doc_counter, res.doc_qty, res.tot_qty = dt
        return res
