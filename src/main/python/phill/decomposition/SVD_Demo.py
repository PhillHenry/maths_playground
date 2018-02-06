import numpy as np
from scipy.sparse.linalg import svds

np.set_printoptions(suppress=True)

X = np.random.normal(size=[5,7])

U, Sigma, VT = svds(X, k=3, tol=0)

print "U", np.shape(U)
print "Sigma", np.shape(Sigma)
print "VT", np.shape(VT)

print "U' U"
print np.dot(U.T, U)

P, D, Q = np.linalg.svd(X, full_matrices=False)
print "P", np.shape(P)
print "D", np.shape(D)
print "Q", np.shape(Q)
