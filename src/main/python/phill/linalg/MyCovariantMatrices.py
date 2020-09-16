from random import gauss

import matplotlib.pyplot as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from os.path import expanduser
import math


def covariance_matrix_from(x):
    # x = original - np.mean(original, axis=0)
    # return np.dot(x, x.T)
    return np.cov(x)
    # return np.dot(x, np.linalg.inv(x))


def lower_left_from(i, j):
    if i >= j:
        return 1
    else:
        return 0


# see https://stephens999.github.io/fiveMinuteStats/normal_markov_chain.html
if __name__ == "__main__":
    lower_left_rows = [lower_left_from(i, j) for i in range(0, 10) for j in range(0, 10)]
    m = np.asmatrix(lower_left_rows).reshape([10, 10])

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    print("M:\n", m)
    c = covariance_matrix_from(m)
    print("c:\n", c)
    bias = np.eye(10) * 0.01
    p = np.linalg.inv(c + bias)
    print("p", p)
