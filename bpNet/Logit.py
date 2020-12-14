from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import stochastic_gradient
from bp.mnist import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
import warnings


def main():
    
    warnings.filterwarnings("ignore")
    print("Start LR Model training...")
    msLog =mnist()
    msTestLog = mnist(kind="t10k")
    Xtrain = msLog.images
    Ytrain = msLog.labels
    Xtest = msTestLog.images
    Ytest = msTestLog.labels
    
    print("Default LR Model loading...")
    start = time.time()
    model = LogisticRegression(solver="liblinear")
    model.fit(Xtrain,Ytrain)
    y_pred = model.predict(Xtest)
    sum=0.0
    for i in range(10000):
        if(y_pred[i]==Ytest[i]):
            sum+=1
    print('Logistics Test set score: %f' % (sum/10000.))
    end = time.time()
    print("Logit Time consuming:%.2fs"%(end-start))#89.78s

    # Simple Tuning Process
    # print("Start param adjusting...")
    # l1test=[]
    # l2test=[]
    # l1 = []
    # l2 = []
    # for i in np.linspace(0.05, 1, 19):
    #     lrl1 = LogisticRegression(penalty="l1", 
    #                               solver = "liblinear", 
    #                               C = i, 
    #                               max_iter=1000)
    #     lrl2 = LogisticRegression(penalty="l2",
    #                               solver = "liblinear", 
    #                               C = i, 
    #                               max_iter=1000)
        
    #     lrl1 = lrl1.fit(Xtrain, Ytrain)
    #     l1.append(accuracy_score(lrl1.predict(Xtrain), Ytrain))
    #     l1test.append(accuracy_score(lrl1.predict(Xtest), Ytest))
        
    #     lrl2 = lrl2.fit(Xtrain, Ytrain)
    #     l2.append(accuracy_score(lrl2.predict(Xtrain), Ytrain))
    #     l2test.append(accuracy_score(lrl2.predict(Xtest), Ytest))
        
    # graph = [l1, l2, l1test, l2test]
    # color = ["green","black","lightgreen","gray"]
    # label = ["L1","L2","L1test", "L2test"]
     
    # plt.figure(figsize=(6,6))
    # for i in range(len(graph)):
    #     plt.plot(np.linspace(0.05,1, 19),graph[i], color[i], label = label[i])
    # plt.legend(loc=4) 

if __name__ == '__main__':
	main()
    
    
