import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log

test_num = 5

red_x = np.random.random(test_num)*5+5
red_y = np.random.random(test_num)*5+5

blue_x = np.random.random(test_num)*5
blue_y = np.random.random(test_num)*5

sum_x = np.hstack((red_x, blue_x))
sum_y = np.hstack((red_y, blue_y))

color = 'k/'*100
color = color.split('/')
color = np.array(color)

k_num = 2

randomcenter_x = np.random.random(k_num)*10
randomcenter_y = np.random.random(k_num)*10

ax1, fig = plt.subplots(figsize=(12,8))
ax1 = plt.scatter(sum_x, sum_y, color = 'k')
ax1 = plt.scatter(randomcenter_x[0], randomcenter_y[0], color = 'r')
ax1 = plt.scatter(randomcenter_x[1], randomcenter_y[1], color = 'b')

plt.show()

for i in range(test_num):
    if ((sum_x[i]-randomcenter_x[0])**2 + (sum_y[i]-randomcenter_y[0])**2) > ((sum_x[i]-randomcenter_x[1])**2 +(sum_y[i]-randomcenter_y[1])**2):
        color[i] = 'r'
    else:
        color[i] = 'b'

    ax2, fig = plt.subplots(figsize=(12, 8))
    ax2 = plt.scatter(sum_x[i], sum_y[i], color=color[i])

ax2 = plt.scatter(randomcenter_x[0], randomcenter_y[0], color = 'r')
ax2 = plt.scatter(randomcenter_x[1], randomcenter_y[1], color = 'b')
plt.show()


