#线性回归（单变量）
#预测food truck的收益值。 ex1data1.txt：数据集，第一列表示城市人数，第二列该城市的food truck收益

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()
# 新增一例，x0
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
Y = data.iloc[:, cols-1:cols]

X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0, 0]))
# 代价函数
def computeCost(X, Y, theta):
    inner = np.power((X * theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))

#正则方程
def normalEqn(X, Y):
    theta = np.linalg.inv(X.T@X)@X.T@Y
    return theta
theta = normalEqn(X, Y)
# 梯度下降的theta为 matrix([[-3.24140214,  1.1272942 ]])
print(theta)

# 梯度下降的cost为 4.515955503078912
computeCost(X, Y, theta.reshape(1, -1))

# 画出拟合图像
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0, 0] + theta[1, 0] * x

plt.figure(figsize=(12, 8))
plt.xlabel('Population')
plt.ylabel('Profit')
l1 = plt.plot(x, f, label='Prediction', color='red')
l2 = plt.scatter(data.Population, data.Profit, label='Traing Data', )
plt.legend(loc='best')
plt.title('Predicted Profit vs Population Size')
plt.show()