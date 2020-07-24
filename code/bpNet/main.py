import os
import numpy as np
from bp.mnist import mnist
from bp.bpModel import bpNet
from bp.testFile import testFile
import time
import matplotlib.pyplot as plt

# 新建mnist数据集对象，初始化数据集
#print("path:", os.path)

"""
# 可调参数：
## 1） hiddenLayersSize=[100, 50] --bpNet
## 2） weightInitStd=0.01         --bpNet
## 3） update函数中的learningrate  --bpNet.update
## 4)  batchsize=100
 
"""

def main(config,fileName = 'data'):
	testFile(fileName)
	Start = time.time()
	msTrain = mnist()
	msTest = mnist(kind="t10k")
	# 提取mnist数据
	trainImg = msTrain.images
	trainLabel = msTrain.oneHotLabels
	testImg = msTest.images
	testLabel = msTest.oneHotLabels
	# 训练数据大小
	trainSize = trainImg.shape[0]
	# 训练批大小
	batchSize = config['batchSize']
	# 迭代次数
	itersNum = config['itersNum']
	# 学习率
	learningRate = config['learningRate']
	# 轮数
	iterPerEpoch = max((trainSize / batchSize), 1)
	# bp初始化
	network = bpNet(hiddenLayersSize=config['hiddenLayersSize'])
	#开始训练模型
	trainA=[]
	testA=[]
	print("iterPerEpoch:",iterPerEpoch)
	#print("Start Model trianing...")
	for index in range(itersNum):
	    if (index % iterPerEpoch == 0):
	        trainAcc = network.accuracy(trainImg, trainLabel)
	        testAcc = network.accuracy(testImg, testLabel)
	        trainA.append(trainAcc)
	        testA.append(testAcc)
	        print("训练精度: %.5f  测试精度: %.5f" % (trainAcc, testAcc))
	    # 获取随机选取的索引
	    trainIndexs = np.random.choice(trainSize, batchSize)
	    imgs = trainImg[trainIndexs]
	    labels = trainLabel[trainIndexs]
	    network.update(imgs, labels, lr=learningRate)
	    
	End = time.time()
	print("BP Time Consuming:%.2f秒"%(End-Start))

	##作图部分
	g = [trainA,testA]
	colors=["blue","orange"]
	labels=["trainAcc","testAcc"]
	plt.figure(figsize=(6,6))
	for i in range(len(g)):
	    plt.plot(g[i], colors[i], label = labels[i])
	    plt.legend(loc=4)

if __name__ == '__main__':

	config = dict(
			batchSize = 100,
			itersNum = 100000,
			learningRate = 0.1,
			hiddenLayersSize = [100,50,50])
    
	main(config)




