import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 代价函数
def computeCost(X, Y, theta):
    inner = np.power((X * theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))

# 梯度下降
def gradientDescent(X, Y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = X * theta.T - Y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = temp[0, j] - alpha / len(X) * np.sum(term)

        theta = temp
        cost[i] = computeCost(X, Y, theta)

    return theta, cost

path = 'ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# 保存mean、std、mins、maxs、data
means = data.mean().values
stds = data.std().values
mins = data.min().values
maxs = data.max().values
data_ = data.values

print('means', means)
print('stds', stds)
print('mins', mins)
print('maxs', maxs)
print('data_', data_)

# 特征缩放 标准化处理
data = (data - data.mean()) / data.std()

# add ones column
data.insert(0, 'Ones', 1)
print(data)

# set X (training data) and Y (target variable)
cols = data.shape[1]
X = data.iloc[:, :cols-1]
Y = data.iloc[:, cols-1:cols]

# 转化为矩阵并初始化theta
X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0, 0, 0]))

# 对数据集进行线性回归
alpha = 0.01
iters = 1000
g, cost = gradientDescent(X, Y, theta, alpha, iters)

# 获取模型的成本（误差）
print(computeCost(X, Y, g))

# 画出iterations-cost图像  直观展示代价函数是否收敛
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Epoch')
plt.show()



#参数转化为缩放前
def theta_transform(theta, means, stds):
    temp = means[:-1] * theta[1:] / stds[:-1]
    theta[0] = (theta[0] - np.sum(temp)) * stds[-1] + means[-1]
    theta[1:] = theta[1:] * stds[-1] / stds[:-1]
    return theta.reshape(1, -1)  #行数固定，列数计算

g_ = np.array(g.reshape(-1, 1))
means = means.reshape(-1, 1)
stds = stds.reshape(-1, 1)  #转化为列向量
transform_g = theta_transform(g_, means, stds)

# 预测价格
def predictPrice(x, y, theta):
    return theta[0, 0] + theta[0, 1]*x + theta[0, 2]*y

# 2104,3,399900,
price = predictPrice(2104, 3, transform_g)
print(price)

# 画出拟合平面
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X_ = np.arange(mins[0], maxs[0]+1, 1)
Y_ = np.arange(mins[1], maxs[1]+1, 1)
X_, Y_ = np.meshgrid(X_, Y_)
Z_ = transform_g[0,0] + transform_g[0,1] * X_ + transform_g[0,2] * Y_

# 手动设置角度
ax.view_init(elev=25, azim=125)

ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

ax.plot_surface(X_, Y_, Z_, rstride=1, cstride=1, color='red')

ax.scatter(data_[:, 0], data_[:, 1], data_[:, 2])
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X_ = np.arange(mins[0], maxs[0]+1, 1)
Y_ = np.arange(mins[1], maxs[1]+1, 1)
X_, Y_ = np.meshgrid(X_, Y_)
Z_ = transform_g[0,0] + transform_g[0,1] * X_ + transform_g[0,2] * Y_

# 手动设置角度
ax.view_init(elev=10, azim=80)

ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

ax.set_xticks(())
ax.set_yticks(())
ax.set_zticks(())
ax.plot_surface(X_, Y_, Z_, rstride=1, cstride=1, color='red')

ax.scatter(data_[:, 0], data_[:, 1], data_[:, 2])
plt.show()
