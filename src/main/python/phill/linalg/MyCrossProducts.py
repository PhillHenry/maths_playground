import numpy as np

# data taken from http://master-studios.net/generalized-convolution-neural-networks/

M = np.matrix('1.0 1.0 0.0;'
              '1.0 1.0 0.0;'
              '0.0 0.0 1.0')

w_A = np.matrix('1.0 0.0 0.0;'
                '0.0 1.0 0.0')

A = np.matrix('1.0 1.0 0.0')

A_e = np.matrix('4.0 5.0 0.0')

w = np.matrix('2.0 1.0')

e = np.matrix('4.0 5.0 1.0').transpose()

print(w_A)
print(A_e)

print("w_A X A\n",np.cross(w_A, A_e))
print("w X A_e\n",np.cross(w, A_e))
print("w X A\n",np.cross(w, A))
print("w X w_A\n",np.cross(w, w_A))
# print("w X M\n",np.cross(w_A, M)) # ValueError: shape mismatch: objects cannot be broadcast to a single shape

