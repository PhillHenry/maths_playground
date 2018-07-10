# import .MonteCarloMain as mc
import os, sys
sys.path.insert(0, os.path.abspath("."))  # https://stackoverflow.com/questions/9427037/relative-path-not-working-even-with-init-py
from MonteCarloMain import l1
from MonteCarloMain import l2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.plot(l1, np.zeros_like(l1) + 0.0, 'x')
    plt.plot(l2, np.zeros_like(l2) + 0.1, 'x')
    plt.axis('off')
    plt.show()