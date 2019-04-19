import matplotlib.pyplot as plt
import numpy as np


def create_matrix(xs, ys):
    X = np.asmatrix(np.array([xs, ys]))
    C = np.dot(X, X.T)
    return X


def eigenvalues_to_matrix(Sigma, U, VT):
    # S = np.diag(Sigma)
    S = np.zeros(shape=[U.shape[1], VT.shape[0]])
    n = 0
    S.itemset((n, n), Sigma[n])
    return S


h = 100
w = 100
n = 500

rxs = np.random.randint(w, size=n)
rys = np.random.randint(h, size=n)

step = 2
sinewave = np.sin(np.linspace(-np.pi, np.pi, (w/step))) * (h / 4)
xs = np.append(w - (np.arange(0, w, step) + sinewave), rxs)
ys = np.append(0.5 * (np.arange(0, h, step) ), rys)

X = create_matrix(xs, ys)

# U, Sigma, VT = svds(X, k=2, tol=0)
U, Sigma, VT = np.linalg.svd(X, full_matrices=False)

print('Eigenvalues = {}'.format(Sigma))

S = eigenvalues_to_matrix(Sigma, U, VT)

print("S =\n{}".format(S))

us = np.dot(U, S)
reconstruction = np.dot(us, VT)
print('X.shape = {}, U.shape = {}, Sigma.shape = {}, S.shape ={}, us.shape = {}, VT.shape = {}, reconstruction = {}'.format(X.shape, U.shape, Sigma.shape, S.shape, us.shape, VT.shape, reconstruction.shape))

to_plot = reconstruction # np.dot(X, VT.transpose())
ptx = np.asarray(to_plot[0, :])[0]
pty = np.asarray(to_plot[1, :])[0]
print('ptx.shape = {}, pty.shape = {}'.format(len(ptx), len(pty)))


fig = plt.figure(0)
fig.add_subplot(121)
plt.scatter(xs, ys, marker="+")
fig.add_subplot(122)
plt.scatter(ptx, pty, marker="+")
plt.show()
print("finished")
