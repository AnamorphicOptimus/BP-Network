# BP-Network

## Introduction

BP-Network is an experimental project that uses BP neural network as the core model to multi-classify MNIST handwritten digit sets. And I realized the construction of BP neural network and the improvement based on the source code through python.
Finally, the improved BP model will be compared with common machine learning and deep learning models, such as random forest and convolutional neural network, to make a comprehensive comparison of model effects and training time.

## Improvement Details:

- 1. Change the activation function from the commonly used sigmoid function to the Relu function<br>
- 2. Secondly, considering that the problem is actually a multi-classification problem, softmax is introduced as the output layer and cross entropy as the loss function<br>
- 3. Finally, batch processing is introduced, and the training set is divided into training batches for training, which improves the operating efficiency of the neural network.<br>

## Experimental Results

| Model | Test Acc | Train Time (s)|
| ------ | ------ | ------ |
| BP | 0.97540 | 35.71 |
| Logistics | 0.92030 | 105.76 |
| SVM | 0.94460 | 935.98 |
| RF | 0.94910 | 5.30 |
| CNN | 0.99200 | 245.98 |

## Experimental Conclusions

In this experiment, the influence of parameter adjustment on the model effect and training time is discussed in detail<br>
The following findings:<br>
- 1. It is found that the training time of the model is more sensitive to the number of iterations and the number of neurons in the hidden layer<br>
- 2. The number of hidden layers, the learning rate and the number of iterations have a greater impact on the training effect<br>
- 3. Compare improved BP model with common machine learning and deep learning models, such as random forest and convolutional neural network, in term of model effects and training time. The result shows that the improved BP neural network has outstanding performance in both aspects. <br>

## Contents
```
.
|-- CNN
|   |-- config.py 							     
|   `-- mnist.py    							
|-- bpNet                                       // BP-Network Source Code
|   |-- Logit.py                                // a separate file for logit regression
|   |-- RandomForest.py                         // a separate file for random forest 
|   |-- SVM.py                                  // a separate file for SVM model
|   |-- bp
|   |   |-- bpModel.py                          // BP layers integration 
|   |   |-- checkFile.py                        // check MNIST data 
|   |   |-- common                              
|   |   |   |-- functions.py                   
|   |   |   `-- layers.py
|   |   `-- mnist.py                            // MNIST data class
|   `-- main.py                                 // BP model run
`-- README.md 

* 7 directories, 26 files 
```
## Operating Instructions

- 1. ./bpNet/main.py: run this file for BP-Network model operation
- 2. ./bpNet/Logit.py, RandomForest.py. SVM.py is for Logit regression, random forest model and SVM respectively
- 3. ./CNN/mnist.py is for CNN model running
