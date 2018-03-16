import unittest

import src.main.python.phill.tf.MySubjects as mg


class MyTestCase(unittest.TestCase):
    docs = ['why hello there', 'omg hello pony', 'she went there? omg']
    words = [w for doc in docs for w in doc.split(' ')]
    print(words)
    targets = [0, 1, 0]
    categories = set(targets)
    n_categories = len(categories)

    def test_max_words(self):
        self.assertEqual(mg.max_words(self.docs), 5)  # grammar counts

    def test_categories(self):
        agg = mg.as_categories(self.docs, self.targets)
        self.assertEqual(len(agg), self.n_categories)

    def test_line_per_category(self):
        cats = mg.line_per_category(self.docs, self.targets)
        print(cats)
        self.assertEqual(len(cats), self.n_categories)

    def test_documents_to_vectors(self):
        vectors = mg.as_vectors_from_dtm(self.docs, self.targets)
        self.assertEqual(len(vectors), len(self.docs))


if __name__ == '__main__':
    unittest.main()
