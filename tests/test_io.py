#!/usr/bin/env python
import unittest
import os
import json

from flexneuart.config import DOCID_FIELD
from flexneuart.io import create_temp_file, open_with_default_enc, FileWrapper, jsonl_gen
from flexneuart.io.json import read_json, save_json
from flexneuart.io.stopwords import read_stop_words
from flexneuart.io.queries import read_queries, write_queries
from flexneuart.io.qrels import read_qrels, write_qrels, QrelEntry
from flexneuart.io.runs import read_run_dict, write_run_dict
from flexneuart.io.train_data import read_pairs_dict, write_pairs_dict

STRING_TO_WRITE = \
    'This is some dummy test\nIt has more than one line\nОн содержит русские буквы!'

STOP_WORDS = ['this', 'that', 'enough_to_test']

QREL_LIST = [QrelEntry(doc_id='1', query_id='1', rel_grade=1), \
                QrelEntry(doc_id='1', query_id='2', rel_grade=1), \
                QrelEntry(doc_id='1', query_id='3', rel_grade=1)]

RUN_DICT = {'1' : {'0' : 1, '1' : 2, '3' : 3},
            '2' : {'0' : 2, '3' : 3, '4' :35}}


class TestIo(unittest.TestCase):
    def test_plain(self):
        tmpfn = create_temp_file()

        with open_with_default_enc(tmpfn, 'w') as f:
            f.write(STRING_TO_WRITE)

        with open_with_default_enc(tmpfn, 'r') as f:
            self.assertEqual(f.read(), STRING_TO_WRITE)

        with FileWrapper(tmpfn, 'r') as f:
            self.assertEqual(f.read(), STRING_TO_WRITE)

        file_lines = []
        with FileWrapper(tmpfn, 'r') as f:
            for line in f:
                file_lines.append(line.rstrip())

        self.assertEqual(file_lines, STRING_TO_WRITE.split('\n'))

        os.unlink(tmpfn)

    def test_compressed(self):
        tmpfn = create_temp_file()
        os.unlink(tmpfn)
        for ext in 'txt', 'gz', 'bz2':
            comp_fn = tmpfn + '.' + ext
            with FileWrapper(comp_fn, 'w') as f:
                f.write(STRING_TO_WRITE)

            with FileWrapper(comp_fn, 'r') as f:
                self.assertEqual(f.read(), STRING_TO_WRITE)

            file_lines = []
            with FileWrapper(comp_fn, 'r') as f:
                for line in f:
                    file_lines.append(line.rstrip())

            self.assertEqual(file_lines, STRING_TO_WRITE.split('\n'))

            os.unlink(comp_fn)

    def test_json_read_write(self):
        tmpfn = create_temp_file()

        save_json(tmpfn, [STRING_TO_WRITE])

        self.assertEqual(read_json(tmpfn), [STRING_TO_WRITE])

        os.unlink(tmpfn)

    def test_jsonl_gen(self):
        tmpfn = create_temp_file()
        os.unlink(tmpfn)

        for ext in 'txt', 'gz', 'bz2':
            comp_fn = tmpfn + '.' + ext

            data = []

            with FileWrapper(comp_fn, 'w') as f:
                for k in range(113):
                    # All keys must be strings
                    elem = {DOCID_FIELD : k, str(k) : f'String {k}', 'bcdf' : k}
                    data.append(elem)
                    elem_s = json.dumps(elem)
                    f.write(elem_s + '\n')

            data_read = list(jsonl_gen(comp_fn))


            self.assertEqual(data, data_read)
            os.unlink(comp_fn)

    def test_read_write_queries(self):
        tmpfn = create_temp_file()
        os.unlink(tmpfn)

        queries = []

        for k in range(113):
            # All keys must be strings
            elem = {DOCID_FIELD : k, str(k) : f'String {k}', 'bcdf' : k}
            queries.append(elem)

        write_queries(queries, tmpfn)

        queries_read = read_queries(tmpfn)

        self.assertEqual(queries, queries_read)
        os.unlink(tmpfn)

    def test_stopwords(self):
        tmpfn = create_temp_file()
        os.unlink(tmpfn)

        with open_with_default_enc(tmpfn, 'w') as f:
            for w in STOP_WORDS:
                f.write(w + '\n')

        self.assertEqual(STOP_WORDS, read_stop_words(tmpfn))

    def test_read_write_qrels(self):
        tmpfn = create_temp_file()
        os.unlink(tmpfn)

        write_qrels(QREL_LIST, tmpfn)

        qrels_read = read_qrels(tmpfn)

        self.assertEqual(qrels_read, QREL_LIST)
        os.unlink(tmpfn)

    def test_read_write_run_dict(self):
        tmpfn = create_temp_file()
        os.unlink(tmpfn)

        write_run_dict(RUN_DICT, tmpfn)

        run_dict_read = read_run_dict(tmpfn)

        self.assertEqual(RUN_DICT, run_dict_read)

        os.unlink(tmpfn)


    def test_read_write_train_pairs(self):
        tmpfn = create_temp_file()
        os.unlink(tmpfn)

        write_pairs_dict(RUN_DICT, tmpfn)

        run_dict_read = read_pairs_dict(tmpfn)

        self.assertEqual(RUN_DICT, run_dict_read)

        os.unlink(tmpfn)



if __name__ == "__main__":
    unittest.main()
