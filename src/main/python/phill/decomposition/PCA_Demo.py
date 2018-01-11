import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# see https://stackoverflow.com/questions/47358216/pca-difference-between-python-numpy-and-sklearn
def eigenfaces(x):
    X = x - np.mean(x, axis = 0) # NB Python seems to be pass by reference
    C = np.dot(X, X.T)
    evals , evecs = np.linalg.eig(C)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    v = np.dot(X.T, evecs)
    v /= np.linalg.norm(v, axis=0)
    # return evals, evecs, C, v
    return v, X, evecs, evals


def plot_3d_matrix(m, ax, col):
    for i in range(np.shape(m)[1]):
        v = m[:, i]
        print "v", v
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
    print "X\n", X
    S = np.diag(s)
    print "u s v\n", np.dot(u, np.dot(S, vt))
    print "X", np.shape(X), "u", np.shape(u), "S", np.shape(S), "vt", np.shape(vt)
    v, x, evecs, evals = eigenfaces(X)
    print "x.v\n", x.dot(v)
    # print "eigenvectors\n", np.shape(evecs)
    print "eigenvalues", evals
    print "v\n", np.shape(v)

    pca = PCA(n_components=3).fit(X)
    res = pca.transform(X)
    print "sklearn\n", res

    # print "vt\n", vt
    # print "X.vt.T.S^-1\n", np.dot(X, np.dot(vt.T, np.linalg.inv(S))) # same as X * vt.T just scaled by S^-1
    # via_svd = np.dot(X, vt.T)
    # print "X.vt.T\n", via_svd
    print "X.u\n", np.dot(X, u)
    # Exactly the same:
    # print "vt.T", vt.T
    # print "vt^-1", np.linalg.inv(vt)

    fig = plt.figure()

    orig_evals, orig_evecs = np.linalg.eig(X)
    ax2 = fig.add_subplot(211, projection='3d')
    plot_3d_matrix(X, ax2, "blue")
    plot_3d_matrix(scaled(orig_evecs, orig_evals), ax2, "red")
    # ax2.text(3, 8, 'eigenvectors', style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10} )
    # ax2.text(3, 8, None, 'eigenvectors', 'italic')
    ax2.set_title('Original')
    ax2.text(0.1, 1, 0.1, 'Eigenvectors',
            transform=ax2.transAxes,
            color='red', fontsize=10)
    ax2.text(0.1, 1, 1, 'Vectors',
            transform=ax2.transAxes,
            color='blue', fontsize=10)

    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('Covariance')
    # ax = Axes3D(fig)
    # plot_3d_matrix(via_svd, ax, "red")

    # plot_3d_matrix(orig_evacs, ax, "grey")
    # plot_3d_matrix(evecs, ax, "green")
    # plot_3d_matrix(u, ax, "grey")

    # plot_3d_matrix(vt.T, ax, "grey")
    # plot_3d_matrix(u, ax, "magenta") # same as X * vt.T

    #  so x.v ~= x.u
    plot_3d_matrix(res, ax, "blue")
    plot_3d_matrix(scaled(evecs, evals), ax, "green")
    plot_3d_matrix(np.dot(x, u), ax, "magenta")  # in the same plane as res


    ax.text(0.1, 1, 0.1, 'Eigenfaces',
             # transform=ax.transAxes,
             color='green', fontsize=10)
    ax.text(0.1, 1, 0.5, 'SVD',
             # transform=ax.transAxes,
             color='magenta', fontsize=10)
    # ax.text(0.1, 1, 13, 'sklearn',
    #          transform=ax2.transAxes,
    #          color='blue', fontsize=10)

    # plot_3d_matrix(np.dot(X, u), ax, "magenta") # same as X * vt.T
    # Conclusion: if we take u,s,VT via SVD of the covariance matrix, C, and the eigenvectors of C (call them v) then C.u = C.v
    # the point is that both SVD and eigendecomposition is on the *covariance* matrix not the original matrix.

    plt.show()
    # Axes3D.plot()

