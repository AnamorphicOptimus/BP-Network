import numpy as np
from bp.common.layers import affineReLu, affine, softmaxLoss


class bpNet:
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

    # Use neural network for prediction
    # This method does not call the output layer, that is, does not call softmax and loss function
    def predict(self, x):
        y = x.copy()
        for layer in self.hiddenLayers:
            y = layer.forward(y)
        return y

    # Calculate the loss function based on the input data and the supervision data
    # an overall data flow will be performed in the neural network
    def loss(self, x, t):
        y = self.predict(x)
        self.lastLayer.forward(y, t)
        return self.lastLayer.loss

    # Backpropagation and Gradient calculation
    def gradient(self, x, t):
        self.loss(x, t)
        d = 1
        d = self.lastLayer.backward(d)
        # Reverse order for derivation of each parameter of the hidden layer
        for index in range(len(self.hiddenLayers) - 1, -1, -1):
            d = self.hiddenLayers[index].backward(d)

    # Update the neural network parameters according to the gradients stored in each layer
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

    # Calculate network prediction accuracy
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if (t.ndim != 1):
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    # region: neural network initialization related methods
    # Build the output layer
    def initOutputLayer(self):
        return softmaxLoss()

    # Construct the hidden layer according to the initial parameter order
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

    # Initialize parameters of all layers in order
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

    # Initialize layer parameters, including weight and bias
    def initLayerParam(self, inputSize, outputSize):
        param = {}
        # Use Gaussian distribution to initialize the weight matrix, here is multiplied by weightInitStd
        param["weight"] = self.weightInitStd * np.random.randn(
            inputSize, outputSize)
        param["bias"] = np.zeros(outputSize)
        return param
