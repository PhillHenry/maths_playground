import numpy as np

def f_softmax(X):
    Z = np.sum(np.exp(X), axis=1)
    # print "\n\nZ =", Z, "\n\n"
    Z = Z.reshape(Z.shape[0], 1)
    # print "\n\nZ =", Z, "\n\n"
    return np.exp(X) / Z

def f_sigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        return f_sigmoid(X)*(1 - f_sigmoid(X))


features = np.matrix('0.0, 0.0;'
                     '1.0, 0.0;'
                     '0.0, 1.0;'
                     '1.0, 1.0')

labels = np.matrix('1.0, 0.0;'
                   '0.0, 1.0;'
                   '0.0, 1.0;'
                   '1.0, 0.0')

weightsLayer1 = np.random.rand(2, 4)

weightsLayer2 = np.random.rand(4, 2)

_0_W = np.zeros([2, 4])
_1_b = np.zeros([1, 2])
_1_W = np.zeros([4, 2])
_0_b = np.zeros([1, 4])


# z = features * weightsLayer1 # TODO add the bias
# print "z:\n", z
# sigmoided = f_sigmoid(z)
#
# print "\nSigmoided:\n", sigmoided
#
# output = sigmoided * weightsLayer2
# print "preOut\n", output
# softmaxed = f_softmax(output)
# print "sofmaxed\n", softmaxed
# delta = softmaxed - labels
# print "grad\n", delta
#
# #OutputLayer.getGradientAndDelta
# weightGradView = np.zeros([4, 2])
#
# weightGradView = (sigmoided.T * delta) + weightGradView
# print "weightGradView:\n", weightGradView
#
# biasGradView = np.sum(delta, 0) # sums columns
# print "biasGradView:\n", biasGradView
#
# # biasGradView and weightGradView both stored in gradient variable
# epsilonNext = (weightsLayer2 * delta.T).T
# print "epsilonNext\n", epsilonNext
#
# _1_W = weightGradView
# _1_b = biasGradView
#
# # in ActivationSigmoid.backprop
# # https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy#
# dLdz = np.multiply(sigmoided, (1-sigmoided))  # f_sigmoid(z, True)
# print "dLdz:\n", dLdz
# # backprop = dLdz * epsilonNext
# backprop = np.multiply(dLdz, epsilonNext)
# print "backprop:\n", backprop
#
# # def f(x):
# #     return 1-x
# # https://stackoverflow.com/questions/7701429/efficient-evaluation-of-a-function-at-every-cell-of-a-numpy-array
# # f1 = np.vectorize(f)
# # print f1(sigmoided)
#
# # DenseLayer.backprop
#
# weightGrad = np.zeros([2, 4])
#
# weightGrad = (features.T * backprop) + weightGrad
# print "weightGrad:\n", weightGrad
#
# biasGrad = np.sum(backprop, 0)
# print "biasGrad:\n", biasGrad
#
# # store biasGrad and weightGrad
#
# epsilonNext2 = (weightsLayer1 * backprop.T).T
# print "epsilonNext2:\n", epsilonNext2
#
# _0_W = weightGrad
# _0_b = biasGrad
#
from math import log
#
# # print log(softmaxed)
# scoreArr = np.multiply(np.vectorize(log)(softmaxed), labels)
# print "scoreArr:\n", scoreArr
#
mini_batch_size = np.shape(features)[0]
# score = -np.sum(scoreArr) / mini_batch_size
# print "score:\n", score # used in BaseOptimzer.checkTerminalConditions
#
# # UpdaterBlock.update
learning_rate = 0.1
# gradView = learning_rate * _0_W
# print "gradView:\n", gradView
#
# # all done in NegativeGradientStepFunction.step as one big array:
# weightsLayer1 = weightsLayer1 - gradView
# print "new weightsLayer1:\n", weightsLayer1
# weightsLayer2 = weightsLayer2 - (learning_rate * _1_W)
# print "new weightsLayer2:\n", weightsLayer2
# biasGrad = biasGrad - (learning_rate * biasGrad)
# biasGradView = biasGradView - (learning_rate * biasGradView)

for i in range(0, 500):
    s0 = features * weightsLayer1
    s0 += _0_b
    sigmoided = f_sigmoid(s0)

    s1 = sigmoided * weightsLayer2
    s1 += _1_b
    softmaxed = f_softmax(s1)

    delta = softmaxed - labels

    epsilonNext = (weightsLayer2 * delta.T).T

    dLdz = np.multiply(sigmoided, (1-sigmoided))
    backprop = np.multiply(dLdz, epsilonNext)


    scoreArr = np.multiply(np.vectorize(log)(softmaxed), labels)
    score = -np.sum(scoreArr) / mini_batch_size
    # print "score:\n", score # used in BaseOptimzer.checkTerminalConditions

    _0_W = (features.T * backprop) + _0_W
    _1_W = (sigmoided.T * delta) + _1_W
    # biasGrad = np.sum(backprop, 0)
    _0_b = - (learning_rate * np.sum(backprop, 0)) + _0_b

    # biasGradView = np.sum(delta, 0) # sums columns
    _1_b = - (learning_rate * np.sum(delta, 0)) + _1_b
    weightsLayer1 = weightsLayer1 - (learning_rate * _0_W)
    weightsLayer2 = weightsLayer2 - (learning_rate * _1_W)


print softmaxed
print labels