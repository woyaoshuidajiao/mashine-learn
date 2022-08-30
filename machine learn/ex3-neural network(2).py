import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.metrics import classification_report
#神经网络

def load_data(path, transpose=True):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    y = y.reshape(y.shape[0])
    print(type(X))
    if transpose:
        X = np.array([im.reshape((20,20)).T.reshape(400) for im in X])
    return X, y

def sigmoid(z):  #准备激活函数
    return 1 / (1 + np.exp(-z))

#前向传播
#已经给出训练得到的theta1,theta2，通过前向传播计算得到预测结果
def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']
theta1, theta2 = load_weight('ex3weights.mat')
print(theta1.shape, theta2.shape)

X, y = load_data('ex3data1.mat', transpose=False)
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
print(X.shape, y.shape)

#输入层
a1 = X
z2 = a1 @ theta1.T
z2 = np.insert(z2, 0, np.ones(z2.shape[0]), axis=1)
print(z2.shape)

#第二层
a2 = sigmoid(z2)
a2.shape
z3 = a2 @ theta2.T
z3.shape

#输出层
a3 = sigmoid(z3)
a3.shape
print(a3)

y_pred = np.argmax(a3, axis=1)+1
print(y_pred)

print(classification_report(y, y_pred))