from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from bp.mnist import mnist
import time
import warnings

def main():
    
    warnings.filterwarnings("ignore")
    print("Start RandomForest Model training...")
    msLog =mnist()
    msTestLog = mnist(kind="t10k")
    Xtrain = msLog.images
    Ytrain = msLog.labels
    Xtest = msTestLog.images
    Ytest = msTestLog.labels
    # n_estimators=10,max_features=math.sqrt(n_features), max_depth=None,min_samples_split=2, bootstrap=True
    print("Default RandomForest model loading...")
    start = time.time()
    clf = RandomForestClassifier()
    clf.fit(Xtrain,Ytrain)
    predict = clf.predict(Xtest)
    score = metrics.accuracy_score(Ytest,predict)
    # Test Accuracy
    report = metrics.accuracy_score(Ytest,predict)
    print(score)
    print(report)
    end = time.time()
    print("RF Time consuming:%.2f秒"%(end-start))

if __name__ == '__main__':

    main()