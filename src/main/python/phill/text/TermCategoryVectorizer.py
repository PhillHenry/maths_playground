import collections

import numpy as np


def from_file(file):
    print("Reading file", file)
    targets =[]
    rows = []
    for line in open(file):
        elements = line.strip().split(",")
        target, *vector = elements
        targets.extend(target)
        rows.append(vector)
    return np.matrix(rows), targets


def matrix_targets(lines, targets):
    size = len(set(targets))
    rows = to_category_vectors(lines, targets, size)
    return np.matrix(rows), targets


def normalize(histo, size):
    xs = [0.] * size
    z = 0.0
    for v in histo.values():
        z = z + v ** 2
    length = z ** 0.5
    for k, v in histo.items():
        xs[k] = v / length
    return xs


def to_category_vectors(docs, cats, size):
    t_to_cs = term_to_cat_dict(zip(docs, cats))
    word_vectors = {}

    for w, h in t_to_cs.items():
        v = normalize(h, size)
        word_vectors[w] = v

    vs = []
    for d in docs:
        vector = np.zeros(size)
        for w in d.split(" "):
            v = np.add(vector, word_vectors[w])
        vs.append(v)

    return vs


def term_to_cat_dict(doc_to_cat):
    """
    :param doc_to_cat: (text, cat id) tuples
    :return: map of term to category histograms
    """
    t_to_cs = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for d, c in doc_to_cat:
        for w in d.split(" "):
            histo = t_to_cs[w]
            x = histo[c]
            y = x + 1
            histo[c] = y
    return t_to_cs
