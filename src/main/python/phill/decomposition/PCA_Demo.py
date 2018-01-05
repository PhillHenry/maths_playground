import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA


# see https://stackoverflow.com/questions/47358216/pca-difference-between-python-numpy-and-sklearn
def eigenfaces(x):
    X = x - np.mean(x, axis = 0)
    C = np.dot(X, X.T)
    evals , evecs = np.linalg.eig(C)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    v = np.dot(X.T, evecs)
    v /= np.linalg.norm(v, axis=0)
    # return evals, evecs, C, v
    return v, X
    # centred = x - np.mean(x, axis=0)
    # covariance_matrix = np.dot(centred, centred.T)
    # evals, evecs = np.linalg.eig(covariance_matrix)
    # idx = np.argsort(evals)[::-1]
    # evecs = evecs[:,idx]
    # evals = evals[idx]
    # v = np.dot(x.T, evecs)
    # v /= np.linalg.norm(v, axis=0)
    # return evals, evecs, centred, v


if __name__ == "__main__":
    X = np.random.rand(3,3)
    u, s, vt = np.linalg.svd(X, full_matrices=0)
    print "X\n", X
    S = np.diag(s)
    print "u s v\n", np.dot(u, np.dot(S, vt))
    print "X", np.shape(X), "u", np.shape(u), "S", np.shape(S), "vt", np.shape(vt)
    v, x = eigenfaces(X)
    print "x.v\n", x.dot(v)
    # print "eigenvectors\n", np.shape(evecs)
    print "v\n", np.shape(v)

    pca = PCA(n_components=3).fit(X)
    res = pca.transform(X)
    print "sklearn\n", res

    # print "vt\n", vt
    print "X.vt.T.S^-1\n", np.dot(X, np.dot(vt.T, np.linalg.inv(S)))
    print "X.vt.T\n", np.dot(X, vt.T)
    # Exactly the same:
    # print "vt.T", vt.T
    # print "vt^-1", np.linalg.inv(vt)


