import os
import numpy as np
from bp.mnist import mnist
from bp.bpModel import bpNet
from bp.checkFile import checkFile
import time
import matplotlib.pyplot as plt

"""
# Tunable Parameter:
## 1） hiddenLayersSize=[100, 50] --bpNet
## 2） weightInitStd=0.01         --bpNet
## 3） learningrate in def:update  --bpNet.update
## 4)  batchsize=100
 
"""

def main(config,fileName = 'data'):
	testFile(fileName)
	Start = time.time()
	msTrain = mnist()
	msTest = mnist(kind="t10k")
	# Extract Mnist data
	trainImg = msTrain.images
	trainLabel = msTrain.oneHotLabels
	testImg = msTest.images
	testLabel = msTest.oneHotLabels
	# Training data size
	trainSize = trainImg.shape[0]
	# Training batch size
	batchSize = config['batchSize']
	# Number of iterations
	itersNum = config['itersNum']
	# Learning rate
	learningRate = config['learningRate']
	# Iter per epoch
	iterPerEpoch = max((trainSize / batchSize), 1)
	# Initialize bp
	network = bpNet(hiddenLayersSize=config['hiddenLayersSize'])
	# start training model
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
	        print("Train Acc: %.5f  Test Acc: %.5f" % (trainAcc, testAcc))
	    # Get randomly selected index
	    trainIndexs = np.random.choice(trainSize, batchSize)
	    imgs = trainImg[trainIndexs]
	    labels = trainLabel[trainIndexs]
	    network.update(imgs, labels, lr=learningRate)
	    
	End = time.time()
	print("BP Time Consuming:%.2fs"%(End-Start))

	# Plot
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




