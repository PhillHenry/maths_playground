from random import gauss

import matplotlib.pyplot as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from os.path import expanduser
import math

# converted from https://github.com/p2t2/figaro/blob/7bdc7c26b011633b4e0a66decc068ffa6f8177f2/FigaroExamples/src/main/scala/com/cra/figaro/example/GaussianProcessTraining.scala


def covariance_ls(i, j, xs, gamma):
    x = xs[i]
    y = xs[j]
    return covariance_fn(x, y, gamma)


def covariance_fn(x, y, gamma):
    return math.exp(- ((x - y) ** 2) * gamma)


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


def variance_of(xs):
    print("variance_of xs = ", xs)
    return np.asmatrix(list(map(lambda x: (x ** 2) + gauss(0, 1), xs)))


def responses_from(xs):
    ys = variance_of(xs)
    print("ys shape = ", np.shape(ys))
    prior = np.sum(ys, axis=0) / len(xs)
    return ys - prior


if __name__ == "__main__":
    xs = np.arange(1, 11).reshape([10, 1])
    responses = responses_from(xs)

    gamma = 1. / 2.0
    C = radial_basis_function_kernel(xs, gamma)
    print("C shape = ", np.shape(C))
    print("C:\n", C)

    bias = np.eye(len(xs)) * 0.001
    print("bias:\n", bias)
    invC = np.linalg.inv(C + bias)  # https://javaagile.blogspot.com/2017/12/ridge-regression.html
    print("C * invC\n", np.dot((C + bias), invC))

    print("invC shape = ", np.shape(invC))
    print("invC:\n", invC)

    print("responses:\n", responses)
    alpha = np.sum(np.multiply(invC, np.transpose(responses)), axis=0)
    print("alpha shape = ", np.shape(alpha))
    print("alpha:\n", alpha)

    pl.figure()
    heat_map(C, "/m_covariance_fn.png")
    heat_map(invC, "/m_inv_covariance_fn.png")
    heat_map(alpha, "/m_alpha.png")

    n = np.size(xs)
    new_co = list(map(lambda x: covariance_fn(x, 7.5, gamma), xs))
    newCovariance = np.asmatrix(new_co).reshape([1, 10])
    new_dot_old = np.dot(newCovariance, invC)
    print("new_dot_old: ", new_dot_old)
    print("newCovariance: ", newCovariance)
    product = np.dot(new_dot_old, np.transpose(newCovariance))
    print("variance: ", np.shape(product))
    variance = 1. - product[0, 0]
    mean_matrix = np.dot(newCovariance, np.transpose(alpha))
    print("mean_matrix = ", mean_matrix)
    mean = mean_matrix[0, 0]
    print("mean = ", mean)
    print("variance = ", variance)
