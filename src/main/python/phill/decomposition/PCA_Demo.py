import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# see https://stackoverflow.com/questions/47358216/pca-difference-between-python-numpy-and-sklearn
def eigenfaces(x):
    X = x - np.mean(x, axis = 0) # NB Python seems to be pass by reference
    C = np.dot(X, X.T)
    # C = np.dot(X.T, X)
    evals , evecs = np.linalg.eig(C)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    v = np.dot(X.T, evecs)
    v /= np.linalg.norm(v, axis=0)
    # return evals, evecs, C, v
    return v, X, evecs, evals, C


def plot_3d_matrix(m, ax, col):
    for i in range(np.shape(m)[1]):
        v = m[:, i]
        # print "v", v
        ax.plot([0, v[0]], [0, v[1]], zs=[0, v[2]], color=col)
        # ax.quiver(0, 0, 0, v[0], v[1], v[2]) # see https://gist.github.com/sytrus-in-github/a3b2ef4414fb144cb08505a060c99b18


def scaled(vectors, values):
    scaled_vectors = np.zeros(np.shape(vectors))
    for i in range(np.shape(vectors)[1]):
        v = vectors[:, i] * values[i]
        scaled_vectors[:, i] = v
    return scaled_vectors


if __name__ == "__main__":
    X = np.random.rand(3,3)
    u, s, vt = np.linalg.svd(X, full_matrices=0)
    # print "X\n", X
    S = np.diag(s)
    # print "u s v\n", np.dot(u, np.dot(S, vt))
    # print "X", np.shape(X), "u", np.shape(u), "S", np.shape(S), "vt", np.shape(vt)
    v, x, evecs, evals, C = eigenfaces(X)
    # print "x.v\n", x.dot(v)
    # print "eigenvalues", evals
    # print "v\n", np.shape(v)

    pca = PCA(n_components=3).fit(X)
    res = pca.transform(X)
    # print "sklearn\n", res

    # print "X.u\n", np.dot(X, u)

    fig = plt.figure()

    orig_evals, orig_evecs = np.linalg.eig(X)
    ax2 = fig.add_subplot(211, projection='3d')
    plot_3d_matrix(X, ax2, "blue")

    scaled_orig_evecs = scaled(orig_evecs, orig_evals)
    plot_3d_matrix(scaled_orig_evecs, ax2, "red")

    plot_3d_matrix(C, ax2, "green")


    ax2.set_title('Original')
    ax2.text(scaled_orig_evecs[0, 0], scaled_orig_evecs[1, 0], scaled_orig_evecs[2, 0], 'Eigenvectors',
            color='red', fontsize=10)
    ax2.text(X[0, 0], X[1, 0], X[2, 0], 'Vectors',
            color='blue', fontsize=10)
    ax2.text(C[0, 0], C[1, 0], C[2, 0], 'Covariance',
            color='green', fontsize=10)

    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('Covariance')

    C_u = np.dot(C, u)
    scaled_evecs = scaled(evecs, evals)
    eFaces = np.dot(C, scaled_evecs)
    plot_3d_matrix(res, ax, "blue")
    plot_3d_matrix(eFaces, ax, "green")
    plot_3d_matrix(C_u, ax, "magenta")  # in the same plane as res

    print "C E\n", eFaces
    print "C U\n", C_u
    print "s\n", s


# https://matplotlib.org/users/text_intro.html
    ax.text(res[0, 0], res[1, 0], res[2, 0], 'sklearn',
             color='green', fontsize=10)
    ax.text(eFaces[0, 0], eFaces[1, 0], eFaces[2, 0], 'Eigenfaces',
             color='blue', fontsize=10)
    ax.text(C_u[0, 0], C_u[1, 0], C_u[2, 0], 'C U',
             color='magenta', fontsize=10)
    # ax.text(0.1, 1, 13, 'sklearn',
    #          transform=ax2.transAxes,
    #          color='blue', fontsize=10)

    # plot_3d_matrix(np.dot(X, u), ax, "magenta") # same as X * vt.T
    # Conclusion: if we take u,s,VT via SVD of the covariance matrix, C, and the eigenvectors of C (call them v) then C.u = C.v
    # the point is that both SVD and eigendecomposition is on the *covariance* matrix not the original matrix.

    plt.show()
    # Axes3D.plot()

