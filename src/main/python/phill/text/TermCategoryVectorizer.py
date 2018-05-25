import collections


def term_to_cat_dict(doc_to_cat):
    t_to_cs = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for d, c in doc_to_cat.items():
        print("d=", d, "c=",c)
        for w in d.split(" "):
            histo = t_to_cs[w]
            x = histo[c]
            y = x + 1
            histo[c] = y
    return t_to_cs
