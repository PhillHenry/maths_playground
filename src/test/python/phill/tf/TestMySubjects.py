import unittest

import src.main.python.phill.tf.MySubjects as mg


class MyTestCase(unittest.TestCase):
    docs = ['why hello there', 'omg hello pony', 'she went there? omg']
    words = [w for doc in docs for w in doc.split(' ')]
    print(words)
    targets = [0, 1, 0]
    categories = set(targets)
    n_categories = len(categories)

    def test_pad_list(self):
        self.assertEqual(len(mg.pad_with_zeros_or_truncate([1] * 3, 5)), 5)

    def test_tuncate_list(self):
        xs = [1] * 5
        truncated = mg.pad_with_zeros_or_truncate(xs, 3)
        self.assertEqual(len(truncated), 3)

    def truncate_first(self):
        xs = [1, 2, 3, 4, 5]
        truncated = mg.pad_with_zeros_or_truncate(xs, 3)
        self.assertEqual(truncated, [1, 2, 3])

    def test_leave_list_the_right_size_alone(self):
        xs = [1] * 5
        self.assertEqual(mg.pad_with_zeros_or_truncate(xs, 5), xs)

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
        n_cat = len(set(self.targets))
        max_w = mg.max_words(self.docs)
        vectors = mg.as_vectors_from_dtm(self.docs, self.targets, n_cat * max_w)
        self.assertEqual(len(vectors), len(self.docs))
        for vec in vectors:
            self.assertEquals(len(vec), n_cat * max_w)


if __name__ == '__main__':
    unittest.main()
