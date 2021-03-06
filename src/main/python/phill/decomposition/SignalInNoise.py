import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def create_large_matrix(xs, ys, w, h):
    X = np.zeros(shape=(h, w))
    for (x, y) in zip(xs, ys):
        # print('x = {}, y ={}'.format(x, y))
        X.itemset((x - 1, y - 1), 1)
    return X


def create_matrix(xs, ys):
    X = np.asmatrix(np.array([xs, ys]))
    return X.transpose()


def eigenvalues_to_matrix(Sigma):
    return np.diag(Sigma)


def top_eigenvalues_to_matrix(Sigma, U, VT, xs):
    S = np.zeros(shape=[U.shape[1], VT.shape[0]])
    for x in xs:
        S.itemset((x, x), Sigma[x])
    return S


def to_ints(xs):
    return map(lambda x: int(x), xs)


def crop(xs, limit):
    return list(filter(lambda x: x > 0 and x < limit, xs))


h = 100
w = 100
n = 500

rxs = np.random.randint(w, size=n)
rys = np.random.randint(h, size=n)

step = 2
distortion = np.sin(np.linspace(-np.pi, np.pi, (w / step))) * (h / 4)
signal_x = crop(0.5 * (w - np.arange(0, w, step)) + distortion, w)
signal_y = np.arange(0, h, step)
xs = np.append(rxs, signal_x)
ys = np.append(rys, signal_y)

X = create_large_matrix(to_ints(ys), to_ints(xs), w, h)
# X = create_matrix(xs, ys)

# U, Sigma, VT = svds(X, k=2, tol=0)
U, Sigma, VT = np.linalg.svd(X, full_matrices=True)

print('Eigenvalues = {}'.format(Sigma))

# S = eigenvalues_to_matrix(Sigma)
ks = range(0, int(Sigma.shape[0] / 4))
S = top_eigenvalues_to_matrix(Sigma, U, VT, ks)

print("S =\n{}".format(S))

us = np.dot(U, S)
reconstruction = np.dot(us, VT)
print('X.shape = {}, U.shape = {}, Sigma.shape = {}, S.shape ={}, us.shape = {}, VT.shape = {}, reconstruction = {}'.format(X.shape, U.shape, Sigma.shape, S.shape, us.shape, VT.shape, reconstruction.shape))

fig = plt.figure(0)
fig.add_subplot(121)
plt.title("Original")
# plt.scatter(xs, ys, marker="+")
plt.imshow(X)
fig.add_subplot(122)

a = reconstruction # np.dot(X, VT.transpose())
# plt.imshow(a, cmap='hot', interpolation='nearest')
# plt.scatter(np.asarray(a[:,0]), np.asarray(a[:,1]), marker="+")
plt.imshow(a) #, cmap='hot', interpolation='nearest')

plt.title("Reconstruction")

plt.show()
print("finished")
