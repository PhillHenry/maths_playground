import numpy as np

if __name__ == "__main__":
    X = np.random.rand(1000, 100)
    u, s, vt = np.linalg.svd(X, full_matrices=0)
    S = np.diag(s)
    print("size of u: ", u.shape)
    print("size of s: ", S.shape)
    print("size of v: ", vt.shape)
