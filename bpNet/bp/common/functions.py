import numpy as np


# The origin softmax function only supports a single dimension. 
# The function of this definition is to take the probability distribution of 0~1
def softmaxOld(x):
    c = np.max(x)
    expA = np.exp(x - c)
    sumExpA = np.sum(expA)
    y = expA / sumExpA
    return y


# Support batch softmax function
def softmax(x):
    cpx = x.copy()
    if (cpx.ndim == 2):
        cpx = cpx.T
        cpx -= np.max(cpx, axis=0)
        y = np.exp(cpx) / np.sum(np.exp(cpx), axis=0)
        return y.T
    else:
        cpx -= np.max(cpx)
        return np.exp(cpx) / np.sum(np.exp(cpx))


# Support batch version of cross entropy function
def crossEntropyError(y, t):
    # For compatible processing, transform one-dimensional data into two-dimensional data
    if (y.ndim == 1):
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # If the output data and the label data size are the same, it is proved to be onehot mode
    # This operation will change onehot mode to non-onthot mode for compatible processing
    if (t.size == y.size):
        # When axis is 1, it is the largest horizontal
        t = t.argmax(axis=1)

    # Get batch size
    batchSize = y.shape[0]
    # np.arange(batch_size) generates a sequence of 0,1,2,3..., which is equivalent to selecting all rows of y
    # t is equivalent to lock to the correct position of each row of label marks and take out that data
    return -np.sum(np.log(y[np.arange(batchSize), t] + 1e-7)) / batchSize
