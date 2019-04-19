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

#plt.scatter(xs, ys, marker="+")
#plt.show()

X = np.zeros(shape=[h, w])
for (x, y) in zip(xs, ys):
    X.itemset((x,y), 1)

# U, Sigma, VT = svds(X, k=2, tol=0)
U, Sigma, VT = np.linalg.svd(X, full_matrices=True)

S = np.diag(Sigma)

# S = np.zeros(shape=[w, h])
# for i in range(len(Sigma)):
#     S.itemset((i, i), Sigma[i])

uv = np.dot(U, S)
all = np.dot(uv, VT)
print('U.shape = {}, Sigma.shape = {}, S.shape ={}, uv.shape = {}, VT.shape = {}, all.shape = {}'.format(U.shape, Sigma.shape, S.shape, uv.shape, VT.shape, all.shape))

x,y = np.argwhere(X != 0).T
print(x)

plt.scatter(x,y)
plt.show()
