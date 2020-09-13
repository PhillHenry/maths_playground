from random import gauss

import matplotlib.pyplot as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# converted from https://github.com/p2t2/figaro/blob/7bdc7c26b011633b4e0a66decc068ffa6f8177f2/FigaroExamples/src/main/scala/com/cra/figaro/example/GaussianProcessTraining.scala

def plot(xs, ys, z, where):
    x, y = np.meshgrid(xs, ys)
    ax = pl.subplot(where, projection='3d')
    ax.plot_surface(x, y, z)


if __name__ == "__main__":
    xs = np.arange(1, 11).reshape([10, 1])
    ys = list(map(lambda x: (x ** 2) + gauss(0, 1), xs))

    print("xs = ", xs)
    print("ys = ", ys)

    X = xs - np.mean(xs, axis=0)

    C = np.dot(X, X.T)

    pl.figure()
    plot(xs, ys, C, 111)
    pl.show()

