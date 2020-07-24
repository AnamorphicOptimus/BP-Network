import numpy as np


# 旧的softmax函数，仅仅支持单个维度，此函数的作用是取0~1的概率分布
def softmaxOld(x):
    c = np.max(x)
    expA = np.exp(x - c)
    sumExpA = np.sum(expA)
    y = expA / sumExpA
    return y


# 支持批处理的softmax函数
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


# 支持批处理版本的交叉熵函数
def crossEntropyError(y, t):
    # 为了兼容处理，变换一维数据为二维数据
    if (y.ndim == 1):
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 如果输出数据和标签数据size一致，则证明为onehot模式
    # 此操作会把onehot模式变为非onthot模式以兼容处理
    if (t.size == y.size):
        # axis为1的时候才是横向取最大
        t = t.argmax(axis=1)

    # 获取批size
    batchSize = y.shape[0]
    # np.arange(batch_size) 生成0,1,2,3...这样的数列，这里相当于选择y的所有行
    # t相当于锁定到每一行label标记的正确位置上，取出那个数据
    return -np.sum(np.log(y[np.arange(batchSize), t] + 1e-7)) / batchSize
