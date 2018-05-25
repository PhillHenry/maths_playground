import unittest

import src.main.python.phill.tf.MySubjects as mg
import numpy as np


class MyTestCase(unittest.TestCase):
    docs = ['why hello there', 'omg hello pony', 'she went there? omg']
    words = [w for doc in docs for w in doc.split(' ')]
    cleaned_words = set(map(lambda x: mg.cleaned(x), words))
    print(words)
    targets = [0, 1, 0]
    categories = set(targets)
    n_categories = len(categories)

    def test_term_class_matrix(self):
        matrix = mg.term_class_matrix(self.docs)
        print("term class matrix\n", matrix)
        self.assertEqual(matrix.shape[0], len(self.docs))
        #self.assertEqual(matrix.shape[1], self.n_categories)


    def test_count_vectorizer(self):
        (matrix, vocab) = mg.matrix_and_vocab(self.docs)
        print("vocabulary = ", vocab)
        print("matrix shape", matrix.shape)
        self.assertEqual(matrix.shape[0], len(self.docs))
        self.assertEqual(matrix.shape[1], len(self.cleaned_words))
        self.assertEqual(set(vocab), set(self.cleaned_words))

    def test_word_to_cat_vector(self):
        word2vec, n_features = mg.word_to_cat_vector(self.docs, self.targets)
        for word in mg.cleaned_docs(self.words):
            v = word2vec[word]
            self.assertEqual(len(v), self.n_categories)

    def test_to_csr(self):
        max_vec_size = mg.max_words(self.docs) * len(self.docs)
        doc_vectors, n_features = mg.to_csr(self.docs, self.targets, max_vec_size)
        self.assertEqual(doc_vectors.shape[0], len(self.docs))
        self.assertEqual(doc_vectors.shape[1], max_vec_size)
        self.assertEqual(n_features, len(set(mg.cleaned_docs(self.words))))

    def test_docs_to_vecs(self):
        max_vec_size = mg.max_words(self.docs) * len(self.docs)
        vecs, n_features = mg.docs_to_vecs(self.docs, self.targets, max_vec_size)
        self.assertEqual(len(vecs), len(self.docs))
        for v in vecs:
            self.assertEqual(len(v), max_vec_size)

    def test_vec_per_category(self):
        cat_to_vec, vec, X = mg.vec_per_category(self.docs, self.targets)
        print(cat_to_vec)
        self.assertEquals(len(cat_to_vec), self.n_categories)
        for i in cat_to_vec:
            row = cat_to_vec[i]
            print("i", i, "row", row)
            self.assertEqual(len(row), len(vec.get_feature_names()))
            self.assertEqual(len(row), len(set(mg.cleaned_docs(self.words))))

    def test_cleaned(self):
        self.assertEqual(mg.cleaned("she went there? omg"), "she went there omg")

    def test_pad_list(self):
        self.assertEqual(len(mg.pad_with_zeros_or_truncate([1] * 3, 5)), 5)

    def test_pad_empty(self):
        xs = []
        truncated = mg.pad_with_zeros_or_truncate(xs, 3)
        self.assertEqual(len(truncated), 3)

    def test_tuncate_list(self):
        xs = [1] * 10
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
