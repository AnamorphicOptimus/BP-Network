import numpy as np
from functions import softmax, crossEntropyError


# Relu layers
class reLu:
    def __init__(self):
        self.mask = None

    # Forward propagation 
    def forward(self, x):
        self.mask = (x <= 0)
        # numpy array copy
        result = x.copy()
        result[self.mask] = 0
        return result

    # Backward propagation
    def backward(self, d):
        result = d.copy()
        result[self.mask] = 0
        return result


# Sigmoid Layers
class sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, d):
        return d * self.y * (1.0 - self.y)


# Affine Layers
class affine:
    def __init__(self, weight, bias):
        self.x = None
        self.weight = weight
        self.bias = bias
        self.xD = None
        self.weightD = None
        self.biasD = None

    def forward(self, x):

        self.x = x
        return np.dot(x, self.weight) + self.bias

    def backward(self, d):
        self.xD = np.dot(d, self.weight.T)
        self.weightD = np.dot(self.x.T, d)
        self.biasD = np.sum(d, axis=0)

        return self.xD



class affineReLu:
    def __init__(self, weight, bias):
        self.affineLayer = affine(weight, bias)
        self.reLuLayer = reLu()

    def forward(self, x):
        y = self.affineLayer.forward(x)
        y = self.reLuLayer.forward(y)
        return y

    def backward(self, d):
        outD = self.reLuLayer.backward(d)
        outD = self.affineLayer.backward(outD)
        return outD

class softmaxLoss:
    def __init__(self):
        self.loss = None
        # outputs of softmax
        self.y = None
        self.t = None

    # Forward propagation: the softmax output and cross entropy are calculated separately
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = crossEntropyError(self.y, self.t)
        return self.y

    def backward(self, dout=1):
        batchSize = self.t.shape[0]
        # is onehot?
        if (self.t.size == self.y.size):
            # Divide by the batch size, and pass a single data error
            # Because the subtraction of two identical matrices is performed, 
            # each derivative value obtained is independent, divided by the batch size
            dx = (self.y - self.t) / batchSize
        else:
            dx = self.y.copy()
            dx[np.arange(batchSize), self.t] -= 1
            dx /= batchSize
        return dx
