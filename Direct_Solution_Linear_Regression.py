# -*- coding: utf-8 -*-
"""
@author: Abdullah Hamza Şahin

Linear Regession with direct solution
"""

import numpy as np
from matplotlib import pyplot as plt

def h(x,w):
    return x*w

X = np.array([31,33,31,49,53,69,101,99,143,132,109])
X = X.reshape(X.shape[0],1)
ones = np.ones((11,1))
X_train = np.concatenate((ones, X), axis=1)

Y = np.array([705,540,650,840,890,850,1200,1150,1700,900,1550])
Y = Y.reshape(Y.shape[0],1)

first_part = np.linalg.inv(np.dot(X_train.transpose(),X_train))
second_part = np.dot(first_part, X_train.transpose())

W = np.dot(second_part, Y)

points = np.arange(0,160,1).reshape(160,1)
ones = np.ones((160,1))

points_ = np.concatenate((ones, points), axis=1)

plt.plot(X,Y, "ro")
plt.axis([0, 160, 0, 1800])
plt.plot(points, np.dot(points_,W))
plt.show()
