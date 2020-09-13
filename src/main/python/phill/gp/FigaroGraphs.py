from random import gauss

import matplotlib.pyplot as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# converted from https://github.com/p2t2/figaro/blob/7bdc7c26b011633b4e0a66decc068ffa6f8177f2/FigaroExamples/src/main/scala/com/cra/figaro/example/GaussianProcessTraining.scala


def plot(xs, ys, z, where, title):
    x, y = np.meshgrid(xs, ys)
    ax = pl.subplot(where, projection='3d', title=title)
    ax.plot_surface(x, y, z)


def covariant_matrix_of(xs):
    X = xs - np.mean(xs, axis=0)
    C = np.dot(X, X.T)
    return C


def covariance_ls(i, j, xs, gamma):
    return ((xs[i] - xs[j]) ** 2) * gamma


def radial_basis_function_kernel(xs, gamma):
    size = len(xs)
    M = [covariance_ls(i, j, xs, gamma) for i in range(0, size) for j in range(0, size)]
    return np.asmatrix(M).reshape([size, size])


if __name__ == "__main__":
    xs = np.arange(1, 11).reshape([10, 1])
    ys = list(map(lambda x: (x ** 2) + gauss(0, 1), xs))

    print("xs = ", xs)
    print("ys = ", ys)

    x = radial_basis_function_kernel(xs, 0.05)
    print("x = ", x)
    C = covariant_matrix_of(x)

    invC = np.linalg.inv(C + (np.eye(len(xs)) * 0.001)) # https://javaagile.blogspot.com/2017/12/ridge-regression.html

    pl.figure()
    plot(xs, ys, x, 311, "Kernel")
    plot(xs, ys, C, 312, "Covariant")
    plot(xs, ys, invC, 313, "Inverse Covariant with bias")
    pl.show()

