import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
print('data.head\n', data.head())
print('\n')
print('data.describe\n', data.describe())

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

#绘制训练集中的样本数据，positive表示接受，negative表示未接受
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
plt.show()

#激活函数
#激活函数的y值分布在[0,1]内，对于分类问题，我们可以使用激活函数的值来表示满足特征的概率。
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

nums = np.arange(-10, 10, step=0.5)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()

#代价函数
def cost(theta, X, Y):
    first = Y * np.log(sigmoid(X@theta.T))
    second = (1 - Y) * np.log(1 - sigmoid(X@theta.T))
    return -1 * np.mean(first + second)


#预处理数据
# add ones column
data.insert(0, 'Ones', 1)
# set X(training data) and Y(target variable)
X = data.iloc[:, 0: -1].values
Y = data.iloc[:, -1].values
theta = np.zeros(3)

#检查矩阵的维度
print(theta)
print(X.shape, Y.shape, theta.shape)

#计算初始的代价（theta=0）
print(cost(theta, X, Y))

#梯度下降
# 计算步长
def gradient(theta, X, Y):
    return (1 / len(X) * X.T @ (sigmoid(X @ theta.T) - Y))

#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     Y = np.matrix(Y)

#     parameters = int(theta.ravel().shape[1])
#     grad = np.zeros(parameters)
#     print(X.shape, theta.shape, (theta.T).shape, (X*theta.T).shape)
#     error = sigmoid(X * theta.T) - Y

#     for i in range(parameters):
#         term = np.multiply(error, X[:, i])
#         grad[i] = np.sum(term) / len(X)

#     return grad
print(gradient(theta, X, Y))
#gradient只是计算了梯度下降theta更新的步长，使用Scipy.optimize.fmin_tnc拟合最优的theta

#拟合参数
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
print(result)
print(type(result))
print(cost(result[0], X, Y))
#使用Scipy.optimize.minimize拟合最优的theta
res = opt.minimize(fun=cost, x0=np.array(theta), args=(X, np.array(Y)), method='Newton-CG', jac=gradient)
print(res)
cost(res.x, X, Y)
print(cost)

#预测分析
def predict(theta, X):
    probability = sigmoid(X @ theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
#模型准确率

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if a^b == 0 else 0 for (a,b) in zip(predictions, Y)]
accuracy = (sum(correct) / len(correct))
print('accuracy = {0:.0f}%'.format(accuracy*100))

# support标签中出现的次数
# precision查准率，recall召回率，f1-score调和平均数
from sklearn.metrics import classification_report
print(classification_report(Y, predictions))

#决策边界
coef = -res.x / res.x[2]
x = np.arange(30, 100, 0.5)
y = coef[0] + coef[1] * x

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.plot(x, y, label='Decision Boundary', c='grey')
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
plt.show()