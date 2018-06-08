import unittest
import os
import src.main.python.phill.text.TermCategoryVectorizer as to_test
import src.main.python.phill.tf.MySubjects as util
import numpy as np


class TestTermCategoryVectorizer(unittest.TestCase):

    doc_to_cat_dict = {"In the beginning": 0, "God created the heavens and the Earth": 0, "In a hole in the ground": 1, "It was the best of times": 2}
    all_docs        = doc_to_cat_dict.keys()
    all_cats        = doc_to_cat_dict.values()
    unique_cats     = set(all_cats)
    unique_words    = set([ws for d in all_docs for ws in d.split(" ")])
    doc2cat         = zip(all_docs, all_cats)

    def test_parse_vector_file(self):
        n_expected_vecs = 3
        matrix, targets = to_test.from_file(os.getcwd() + "/../../../resources/term_category.csv")
        self.assertEqual(len(targets), n_expected_vecs)
        self.assertEqual(matrix.shape[0], n_expected_vecs)

    def test_matrix_targets(self):
        (matrix, targets) = to_test.matrix_targets(self.all_docs, self.all_cats)
        self.assertEqual(matrix.shape[0], len(targets))

    def test_normalize(self):
        n = 10
        vector = {1: 5, 2: 4, 7: 6}
        vec = to_test.normalize(vector, n)
        self.assertEqual(len(vec), n)
        self.assert_is_normalized(vec)

    def test_docs_to_vectors(self):
        vectors = to_test.to_category_vectors(self.all_docs, self.all_cats, len(self.unique_cats))
        self.assertEqual(len(vectors), len(self.all_docs))
        for vec in vectors:
            self.assertEqual(len(vec), len(self.unique_cats))
            x = np.asarray(vec)
            print("x=", x)
            self.assert_is_normalized(x)

    def test_terms_to_cats(self):
        t2c = to_test.term_to_cat_dict(zip(self.all_docs, self.all_cats))
        keys = [k for k, v in t2c.items()]
        print(keys)
        self.assertEqual(len(t2c), len(self.unique_words))
        self.assertEqual(set(keys), self.unique_words)

    def assert_is_normalized(self, xs):
        self.assertAlmostEqual(sum(map(lambda x: x**2, xs)), 1.0)
