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
data.head()

cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
Y = data.iloc[:, cols-1:cols]

X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0, 0]))
#scikit-learn
from sklearn import linear_model
model = linear_model.LinearRegression()
print(model.fit(X, Y))

x = np.array(X[:, 1].A1)
y = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, y, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()