# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:49:47 2020
!!!!!!IMPORTANT!!!!!!!!
Each part seperated from each other. Please run cell by cell
@author: Abdullah Hamza Şahin
"""
import numpy as np
from matplotlib import pyplot as plt

def calculateEmpiricalRisk(w0, w, x, y):
    totalError = 0
    for i in range(0, len(y)):
        totalError += (y[i] - (w * x[i] + w0)) ** 2
    return totalError / float(len(y))

def gradientDescent(w0, w, X, Y, learningRate):
    w0_gradient = 0
    w_gradient = 0
    m = len(Y)
    for i in range(0, m):
        x = X[i].reshape(1,1)
        y = Y[i].reshape(1,1)
        w0_gradient += -(2/m) * (y - ((w * x) + w0))
        w_gradient += -(2/m) * x * (y - ((w * x) + w0))
    new_w0 = w0 - (learningRate * w0_gradient)
    new_w = w - (learningRate * w_gradient)
    return [new_w0, new_w]



X = np.array([31,33,31,49,53,69,101,99,143,132,109])
X = X.reshape(X.shape[0],1)

Y = np.array([705,540,650,840,890,850,1200,1150,1700,900,1550])
Y = Y.reshape(Y.shape[0],1)
#%%
#Step size of 10^(-5)
iteration = 40
lr = 1*10**(-5)
W = np.array([100])
W0 = 100
risk = []
for i in range(iteration):
    W0, W = gradientDescent(W0, W, X, Y, lr)
    risk.append(calculateEmpiricalRisk(W0, W, X, Y)[0])
    if i % 5 == 0:
        plt.plot(X, Y, "ro")
        plt.axis([0, 160, 0, 1800])
        plt.plot(X, np.dot(X,W)+W0)
        plt.show()
        plt.close()
        
plt.plot(range(iteration), risk)
#%%
#Step size of 10^(-4)
iteration = 15
lr = 1*10**(-4)
W = np.array([100])
W0 = 100
risk = []
for i in range(iteration):
    W0, W = gradientDescent(W0, W, X, Y, lr)
    risk.append(calculateEmpiricalRisk(W0, W, X, Y)[0])
    if i % 1 == 0:
        plt.plot(X, Y, "ro")
        plt.axis([0, 160, 0, 1800])
        plt.plot(X, np.dot(X,W)+W0)
        plt.show()
        plt.close()
        print("Empirical Risk for iterariton ",i,":",calculateEmpiricalRisk(W0, W, X, Y))
    
plt.plot(range(iteration), risk)

