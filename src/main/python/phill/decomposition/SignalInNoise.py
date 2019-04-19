import numpy as np
from scipy.sparse.linalg import svds
import numpy as np
import matplotlib.pyplot as plt

h = 100
w = 100
n = 500

rxs = np.random.randint(w, size=n)
rys = np.random.randint(h, size=n)

xs = np.append(np.arange(0, h), rxs)
ys = np.append(np.arange(0, h), rys)

X = np.asmatrix(np.array([xs, ys]))

# U, Sigma, VT = svds(X, k=2, tol=0)
U, Sigma, VT = np.linalg.svd(X, full_matrices=True)

# S = np.diag(Sigma)

S = np.zeros(shape=[U.shape[1], VT.shape[0]])

S.itemset((0, 0), Sigma[0])

uv = np.dot(U, S)
all = np.dot(uv, VT)
print('U.shape = {}, Sigma.shape = {}, S.shape ={}, uv.shape = {}, VT.shape = {}, all = {}'.format(U.shape, Sigma.shape, S.shape, uv.shape, VT.shape, all.shape))

# x,y = np.argwhere(X != 0).T
ptx = np.asarray(all[0, :])[0]
pty = np.asarray(all[1, :])[0]
print('ptx.shape = {}, pty.shape = {}'.format(len(ptx), len(pty)))


fig = plt.figure(0)
fig.add_subplot(121)
plt.scatter(xs, ys, marker="+")
fig.add_subplot(122)
plt.scatter(ptx, pty)
plt.show()
print("finished")
