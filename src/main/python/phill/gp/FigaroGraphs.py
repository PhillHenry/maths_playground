from random import gauss

import matplotlib.pyplot as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from os.path import expanduser
import math

# converted from https://github.com/p2t2/figaro/blob/7bdc7c26b011633b4e0a66decc068ffa6f8177f2/FigaroExamples/src/main/scala/com/cra/figaro/example/GaussianProcessTraining.scala


def plot(xs, ys, z, where, title):
    x, y = np.meshgrid(xs, ys)
    ax = pl.subplot(where, projection='3d', title=title)
    ax.plot_surface(x, y, z)


def covariance_ls(i, j, xs, gamma):
    return math.exp(- ((xs[i] - xs[j]) ** 2) * gamma)


def radial_basis_function_kernel(xs, gamma):
    size = len(xs)
    m = [covariance_ls(i, j, xs, gamma) for i in range(0, size) for j in range(0, size)]
    return np.asmatrix(m).reshape([size, size])


def heat_map(m, file):
    dir = expanduser("~") + "/Pictures/"
    heatmap = pl.imshow(m, cmap='hot', interpolation='nearest')
    pl.colorbar(heatmap)
    pl.savefig(dir + "/" + file)
    pl.clf()


if __name__ == "__main__":
    xs = np.arange(1, 11).reshape([10, 1])
    ys = np.asmatrix(list(map(lambda x: (x ** 2) + gauss(0, 1), xs)))
    print("ys shape = ", np.shape(ys))

    C = radial_basis_function_kernel(xs, 1. / 2.0)
    print("C shape = ", np.shape(C))
    print("C:\n", C)

    bias = np.eye(len(xs)) * 0.001
    print("bias:\n", bias)
    invC = np.linalg.inv(C + bias)  # https://javaagile.blogspot.com/2017/12/ridge-regression.html
    print("C * invC\n", np.dot((C + bias), invC))

    print("invC shape = ", np.shape(invC))
    print("invC:\n", invC)

    alpha = np.dot(invC, ys)
    print("alpha shape = ", np.shape(alpha))
    print("alpha:\n", alpha)

    pl.figure()
    heat_map(C, "/m_covariance_fn.png")
    heat_map(invC, "/m_inv_covariance_fn.png")
    heat_map(alpha, "/m_alpha.png")

