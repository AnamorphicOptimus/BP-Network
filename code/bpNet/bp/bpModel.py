import numpy as np
from bp.common.layers import affineReLu, affine, softmaxLoss


class bpNet:
    # 构造函数
    def __init__(
        self,
        inputSize=784,
        outputSize=10,
        hiddenLayersSize=[100, 50],
        weightInitStd=0.01,
    ):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayersSize = hiddenLayersSize
        self.weightInitStd = weightInitStd

        self.params = self.initParams()
        self.hiddenLayers = self.initHiddenLayers()
        self.lastLayer = self.initOutputLayer()

    # 使用神经网络进行预测
    # 此方法没有调用输出层，即没有调用softmax以及损失函数
    def predict(self, x):
        y = x.copy()
        for layer in self.hiddenLayers:
            y = layer.forward(y)
        return y

    # 根据输入数据以及监督数据计算损失函数
    # 同时也会对神经网络进行一次整体数据流动
    def loss(self, x, t):
        y = self.predict(x)
        self.lastLayer.forward(y, t)
        return self.lastLayer.loss

    # 反向传播计算梯度
    def gradient(self, x, t):
        self.loss(x, t)
        d = 1
        d = self.lastLayer.backward(d)
        # 倒序对于隐藏层的各个参数求导
        for index in range(len(self.hiddenLayers) - 1, -1, -1):
            d = self.hiddenLayers[index].backward(d)

    # 根据保存在各层的梯度更新神经网络参数
    def update(self, x, t, lr=0.1):
        self.gradient(x, t)
        for layer in self.hiddenLayers:
            weight = None
            bias = None
            if (isinstance(layer, affineReLu)):
                layer.affineLayer.weight -= layer.affineLayer.weightD * lr
                layer.affineLayer.bias -= layer.affineLayer.bias * lr
            elif (isinstance(layer, affine)):
                layer.weight -= layer.weightD * lr
                layer.bias -= layer.biasD * lr

    # 计算网络预测精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if (t.ndim != 1):
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    # region 神经网络初始化相关方法
    # 构建输出层
    def initOutputLayer(self):
        return softmaxLoss()

    # 根据初始化的参数顺序构建隐藏层
    def initHiddenLayers(self):
        layers = []
        for index, value in enumerate(self.params):
            weight = value["weight"]
            bias = value["bias"]
            layer = None
            if (index == len(self.params) - 1):
                layer = affine(weight, bias)
            else:
                layer = affineReLu(weight, bias)
            layers.append(layer)
        return layers

    # 顺序初始化各层参数
    def initParams(self):
        params = []
        layerSizeList = [self.inputSize
                         ] + self.hiddenLayersSize + [self.outputSize]
        for index, value in enumerate(layerSizeList):
            if (index > 0):
                prevSize = layerSizeList[index - 1]
                curSize = value
                param = self.initLayerParam(prevSize, curSize)
                params.append(param)
        return params

    # 初始化层参数，包括权重和偏置
    def initLayerParam(self, inputSize, outputSize):
        param = {}
        # 利用高斯分布初始化权重矩阵，这里乘以了weightInitStd
        param["weight"] = self.weightInitStd * np.random.randn(
            inputSize, outputSize)
        param["bias"] = np.zeros(outputSize)
        return param
