from sklearn import svm,metrics
from bp.mnist import mnist
import warnings
import time

def main():
    warnings.filterwarnings("ignore")
    print("Start SVM Model loading...")
    msSVM = mnist()
    msTestSVM = mnist(kind="t10k")
    Xtrain = msSVM.images
    Ytrain = msSVM.labels
    Xtest = msTestSVM.images
    Ytest = msTestSVM.labels
    
    # Initialize SVM
    print("Default SVM Model loading...")
    start = time.time()
    clf=  svm.SVC()
    clf.fit(Xtrain,Ytrain)
    predict = clf.predict(Xtest)
    score = metrics.accuracy_score(Ytest,predict)
    # Test Acc
    report = metrics.accuracy_score(Ytest,predict)
    end = time.time()
    print("SVM Time consuming:%.2f秒"%(end-start))
    print(score)
    print(report)

if __name__ == '__main__':

    main()

