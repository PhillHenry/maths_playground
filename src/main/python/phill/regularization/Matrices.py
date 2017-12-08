import numpy as np
from numpy.linalg import matrix_rank, det, inv


mat_rank_1 = np.matrix('1    2    3;'
                       '10   20   30;'
                       '100, 200, 300')

print mat_rank_1, "has rank ", matrix_rank(mat_rank_1)  # it is indeed 1

mmt = np.dot(mat_rank_1, mat_rank_1.T)

print "\ndeterminant of rank 1 matrix multiplied by its transpose:", det(mmt)  # and not too surprisingly, it's 0

# this blows up with "numpy.linalg.linalg.LinAlgError: Singular matrix"
# inv(mmt)

l = np.eye(3, 3) * 0.1

with_bias = mat_rank_1 + l

wb_wbT = np.dot(with_bias, with_bias.T)
print wb_wbT, "has determinant", det(wb_wbT), "and rank", matrix_rank(wb_wbT)

print inv(wb_wbT)
