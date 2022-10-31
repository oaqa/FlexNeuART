#!/usr/bin/env python
import unittest

from flexneuart.eval import get_cutoff

class TestCutoff(unittest.TestCase):
    def test1(self):
        self.assertEqual(get_cutoff('mrr', 'mrr'), float('inf'))
        self.assertEqual(get_cutoff('mrr', 'mrr@20'), 20)
        self.assertEqual(get_cutoff('ndcg', 'ndcg@130'), 130)
