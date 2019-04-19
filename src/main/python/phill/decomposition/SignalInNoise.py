import matplotlib.pyplot as plt
import numpy as np


def createMatrix(xs, ys):
    X = np.asmatrix(np.array([xs, ys]))
    C = np.dot(X, X.T)
    return X

def eigenvalues_to_matrix(Sigma, U, VT):
    S = np.diag(Sigma)
    # S = np.zeros(shape=[U.shape[1], VT.shape[0]])
    # S.itemset((0, 0), Sigma[0])
    return S


h = 100
w = 100
n = 500

rxs = np.random.randint(w, size=n)
rys = np.random.randint(h, size=n)

step = 1
xs = np.append(w - np.arange(0, h, step), rxs)
ys = np.append(0.5 * np.arange(0, h, step), rys)

X = createMatrix(xs, ys)

# U, Sigma, VT = svds(X, k=2, tol=0)
U, Sigma, VT = np.linalg.svd(X, full_matrices=False)

S = eigenvalues_to_matrix(Sigma, U, VT)

uv = np.dot(U, S)
reconstruction = np.dot(uv, VT)
print('X.shape = {}, U.shape = {}, Sigma.shape = {}, S.shape ={}, uv.shape = {}, VT.shape = {}, reconstruction = {}'.format(X.shape, U.shape, Sigma.shape, S.shape, uv.shape, VT.shape, reconstruction.shape))

# x,y = np.argwhere(X != 0).T
ptx = np.asarray(reconstruction[0, :])[0]
pty = np.asarray(reconstruction[1, :])[0]
print('ptx.shape = {}, pty.shape = {}'.format(len(ptx), len(pty)))


fig = plt.figure(0)
fig.add_subplot(121)
plt.scatter(xs, ys, marker="+")
fig.add_subplot(122)
plt.scatter(ptx, pty, marker="+")
plt.show()
print("finished")
