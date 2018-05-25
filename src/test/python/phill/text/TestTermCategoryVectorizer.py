import unittest

import src.main.python.phill.text.TermCategoryVectorizer as to_test
import src.main.python.phill.tf.MySubjects as util

class TestTermCategoryVectorizer(unittest.TestCase):

    doc_to_cat      = {"In the beginning": 1, "God created the heavens and the Earth": 1, "In a hole in the ground": 2, "It was the best of times": 3}
    all_docs        = doc_to_cat.keys()
    all_cats        = doc_to_cat.values()
    unique_cats     = set(all_cats)
    #unique_words    = set([util.cleaned(ws) for d in all_docs for ws in d.split(" ")])
    unique_words    = set([ws for d in all_docs for ws in d.split(" ")])
    print("unique words: ", unique_words)

    def test_terms_to_cats(self):
        print("docs", self.doc_to_cat)
        t2c = to_test.term_to_cat_dict(self.doc_to_cat)
        keys = [k for k, v in t2c.items()]
        print(keys)
        self.assertEqual(len(t2c), len(self.unique_words))
        self.assertEqual(set(keys), self.unique_words)
