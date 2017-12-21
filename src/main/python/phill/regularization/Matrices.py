import numpy as np
from numpy.linalg import matrix_rank, det, inv, norm


mat_rank_1 = np.matrix('1    2    3;'
                       '10   20   30;'
                       '100, 200, 300')

print mat_rank_1, "has rank ", matrix_rank(mat_rank_1)  # it is indeed 1

mTm = np.dot(mat_rank_1, mat_rank_1)

print "\ndeterminant of rank 1 matrix multiplied by its transpose:", det(mTm)  # and not too surprisingly, it's 0

# this blows up with "numpy.linalg.linalg.LinAlgError: Singular matrix"
# inv(mmt)

l = np.eye(3, 3) * 0.1

with_bias = mat_rank_1 + l

wbTwb = np.dot(mat_rank_1.T, mat_rank_1) + l
print wbTwb, "has determinant", det(wbTwb), "and rank", matrix_rank(wbTwb)

print inv(wbTwb)

print "norm of original matrix", norm(with_bias)
print "norm of inverse of original matrix", norm(inv(with_bias))
print "Condition number", norm(with_bias) * norm(inv(with_bias))
print
print "norm of original matrix", norm(wbTwb)
print "norm of inverse of original matrix", norm(inv(wbTwb))
print "Condition number", norm(wbTwb) * norm(inv(wbTwb))
