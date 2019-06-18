import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edges_from([(1,2),(2,3),(1,3),(2,4)])
labelmap = dict(zip(G.nodes(), ["A", "B", "C", "D"]))
nx.draw(G, labels=labelmap, with_labels=True)
plt.show()
