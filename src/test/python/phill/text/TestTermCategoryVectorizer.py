import unittest

import src.main.python.phill.text.TermCategoryVectorizer as to_test
import src.main.python.phill.tf.MySubjects as util
import numpy as np


class TestTermCategoryVectorizer(unittest.TestCase):

    doc_to_cat      = {"In the beginning": 0, "God created the heavens and the Earth": 0, "In a hole in the ground": 1, "It was the best of times": 2}
    all_docs        = doc_to_cat.keys()
    all_cats        = doc_to_cat.values()
    unique_cats     = set(all_cats)
    unique_words    = set([ws for d in all_docs for ws in d.split(" ")])

    def test_normalize(self):
        n = 10
        vector = {1: 5, 2: 4, 7: 6}
        vec = to_test.normalize(vector, n)
        self.assertEqual(len(vec), n)
        self.is_normal(vec)

    def test_docs_to_vectors(self):
        vectors = to_test.to_category_vectors(self.doc_to_cat)
        self.assertEqual(len(vectors), len(self.all_docs))
        for vec in vectors:
            self.assertEqual(len(vec), len(self.unique_cats))
            x = np.asarray(vec)
            print("x=", x)
            self.is_normal(x)

    def test_terms_to_cats(self):
        print("docs", self.doc_to_cat)
        t2c = to_test.term_to_cat_dict(self.doc_to_cat)
        keys = [k for k, v in t2c.items()]
        print(keys)
        self.assertEqual(len(t2c), len(self.unique_words))
        self.assertEqual(set(keys), self.unique_words)

    def is_normal(self, xs):
        self.assertAlmostEqual(sum(map(lambda x: x**2, xs)), 1.0)
